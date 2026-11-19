#!/usr/bin/env ruby
# frozen_string_literal: true

# Download every file of a Hugging Face repo page, resumably and verified.
#
#     ruby hf_download.rb https://huggingface.co/zai-org/GLM-5.3-Flash/tree/main
#
# Each page gets its own folder (here: ./zai-org__GLM-5.3-Flash). Just run the
# same command again after a crash or Ctrl-C: finished files are skipped and
# partial files continue from the last byte on disk.
#
# How it stays safe:
#   * Data is streamed into "<name>.<hash-prefix>.part" and only renamed to the
#     real file name after its checksum matched, so a file with its final name
#     is always complete and verified.
#   * Checksums come from the Hub's file listing: SHA-256 for LFS files, the git
#     blob SHA-1 for small regular files.
#   * The part name contains the expected hash, so if the repo is updated
#     between runs a stale partial is never mixed with the new version.
#   * .hfdl-manifest.json remembers verified files (hash, size, mtime) so
#     restarts don't re-hash everything. Use --verify to re-hash anyway.
#
# Once everything is downloaded, the folder is turned into a git clone of the
# repo (needs git; --nogit skips this, --togit converts an earlier download).
# Only the small git history is fetched for that - the large files are the ones
# already on disk, and they are neither downloaded nor hashed a second time.
#
# Apart from git, only the Ruby standard library is needed. For gated/private
# repos set HF_TOKEN (or log in with `huggingface-cli login`).
#
# This is a port of python/hf_download.py. Both versions use the same folder
# names, part files and manifest, so either can resume the other's download.

require "digest"
require "fileutils"
require "io/console"
require "json"
require "net/http"
require "open3"
require "openssl"
require "optparse"
require "uri"

ENDPOINT = "https://huggingface.co"
CHUNK = 1 << 20
TIMEOUT = 60 # seconds without data before a connection counts as stalled
MANIFEST = ".hfdl-manifest.json"
NOW_WINDOW = 3      # seconds of history behind the "current" speed
AVG_WINDOW = 300    # seconds of history behind the average speed

# An error that retrying will not fix.
class Fatal < StandardError; end

# A response with an error status.
class HTTPError < StandardError
  attr_reader :code

  def initialize(code, reason)
    @code = code
    super("HTTP Error #{code}: #{reason}")
  end
end

# Errors that are worth another attempt.
NETWORK_ERRORS = [SystemCallError, IOError, Timeout::Error, SocketError, OpenSSL::SSL::SSLError,
                  Net::ProtocolError, Net::HTTPBadResponse, Net::HTTPHeaderSyntaxError].freeze

# Set once on Ctrl-C; workers check it between chunks.
class StopFlag
  def initialize
    @set = false
    @mutex = Mutex.new
    @cond = ConditionVariable.new
  end

  def set
    @mutex.synchronize do
      @set = true
      @cond.broadcast
    end
  end

  def set?
    @set
  end

  # Sleep for `seconds`, or less if the flag gets set meanwhile.
  def wait(seconds)
    @mutex.synchronize { @cond.wait(@mutex, seconds) unless @set }
  end
end

STOP = StopFlag.new

def mono
  Process.clock_gettime(Process::CLOCK_MONOTONIC)
end

# ---------------------------------------------------------------- HTTP

# Percent-encode everything but unreserved characters and those in `safe`.
def quote(str, safe = "/")
  str.b.gsub(/[^A-Za-z0-9_.\-~#{Regexp.escape(safe)}]/n) { |c| format("%%%02X", c.ord) }
     .force_encoding(Encoding::UTF_8)
end

def unquote(str)
  str.b.gsub(/%([0-9A-Fa-f]{2})/n) { Regexp.last_match(1).hex.chr }.force_encoding(Encoding::UTF_8)
end

def find_token
  %w[HF_TOKEN HUGGING_FACE_HUB_TOKEN].each do |var|
    value = ENV[var]
    return value.strip if value && !value.empty?
  end
  home = ENV["HF_HOME"]
  home = File.expand_path("~/.cache/huggingface") if home.nil? || home.empty?
  begin
    token = File.read(File.join(home, "token")).strip
    token.empty? ? nil : token
  rescue SystemCallError
    nil
  end
end

# GET `url`, following redirects, and yield the successful response so the
# caller can stream its body. Returns the block's value; raises HTTPError for
# an error status.
def http_get(url, token, offset = 0)
  auth = token
  20.times do
    uri = URI(url)
    req = Net::HTTP::Get.new(uri)
    req["User-Agent"] = "hf-download-script/1.0"
    req["Accept-Encoding"] = "identity"
    req["Authorization"] = "Bearer #{auth}" if auth
    req["Range"] = "bytes=#{offset}-" if offset.positive?

    http = Net::HTTP.new(uri.host, uri.port)
    http.use_ssl = uri.scheme == "https"
    http.open_timeout = TIMEOUT
    http.read_timeout = TIMEOUT
    # Net::HTTP would silently repeat a GET that fails half-way and hand the
    # block a second response from the start. Retries are handled by the caller.
    http.max_retries = 0

    location = nil
    http.start do
      http.request(req) do |res|
        case res
        when Net::HTTPRedirection
          location = res["location"]
        when Net::HTTPSuccess
          return yield(res)
        else
          raise HTTPError.new(res.code.to_i, res.message)
        end
      end
    end
    raise HTTPError.new(0, "redirect without a Location header") unless location

    new_url = URI.join(url, location)
    # Don't leak the HF token to the CDN (which also rejects it).
    auth = nil if [new_url.host, new_url.port] != [uri.host, uri.port]
    url = new_url.to_s
  end
  raise HTTPError.new(0, "too many redirects")
end

# GET a JSON document, retrying transient errors. Returns [data, response].
def get_json(url, token)
  6.times do |attempt|
    err = nil
    begin
      return http_get(url, token) { |res| [JSON.parse(res.read_body), res] }
    rescue HTTPError => e
      if [401, 403, 404].include?(e.code)
        raise Fatal, "HTTP #{e.code} for #{url} - wrong URL, or a gated/private " \
                     "repo (set HF_TOKEN and accept its terms on the website)"
      end
      err = e
    rescue *NETWORK_ERRORS, JSON::ParserError => e
      err = e
    end
    raise Fatal, "could not fetch #{url}: #{err.message}" if attempt == 5

    sleep(2**attempt)
  end
end

# ---------------------------------------------------------------- repo listing

RepoFile = Struct.new(:path, :size, :algo, :digest) do
  def new_hasher
    return Digest::SHA256.new if algo == "sha256"

    Digest::SHA1.new.update("blob #{size}\0") # git blob id
  end
end

HFS_IGNORABLE = "‌‍‎‏﻿‪-‮⁪-⁯"

# True if a path component could be taken for ".git" by some filesystem.
#
# Covers case-insensitive names, the Unicode characters HFS+ ignores, and the
# NTFS forms (trailing dots/spaces, the short name GIT~1, and a ":stream"
# suffix as in ".git::$INDEX_ALLOCATION") - the same cases git itself
# refuses to check out.
def dotgit?(name)
  name = (name.split(":", 2)[0] || "").delete(HFS_IGNORABLE).downcase.sub(/[ .]+\z/, "")
  name == ".git" || name.match?(/\Agit~\d+\z/)
end

# -> [kind, repo_id, revision, subfolder]
def parse_url(url)
  url = "#{ENDPOINT}/#{url.sub(%r{\A/+}, '')}" unless url.include?("://")
  path = url.sub(%r{\A[^:/?#]+://[^/?#]*}, "").sub(/[?#].*\z/m, "")
  parts = path.split("/").reject(&:empty?).map { |p| unquote(p) }
  kind = "models"
  kind = parts.shift if %w[datasets spaces models].include?(parts[0])
  rev = "main"
  sub = []
  %w[tree blob resolve].each do |marker|
    next unless parts.first(3).include?(marker)

    i = parts.index(marker)
    rev = parts[i + 1] if parts.length > i + 1
    sub = marker == "tree" ? (parts[(i + 2)..] || []) : []
    parts = parts[0...i]
    break
  end
  if !parts.length.between?(1, 2) || parts.any? { |p| [".", ".."].include?(p) }
    raise Fatal, "cannot find a repo name in #{url.inspect}"
  end

  [kind, parts.join("/"), rev, sub.join("/")]
end

# Pin the revision to a commit and list its files. -> [commit, [RepoFile]]
def list_files(kind, repo, rev, sub, token)
  info, = get_json("#{ENDPOINT}/api/#{kind}/#{quote(repo)}/revision/#{quote(rev, '')}", token)
  commit = info["sha"]
  url = "#{ENDPOINT}/api/#{kind}/#{quote(repo)}/tree/#{commit}"
  url += "/#{quote(sub)}" unless sub.empty?
  url += "?recursive=true"
  files = []
  while url
    entries, res = get_json(url, token)
    entries.each do |e|
      next unless e["type"] == "file"

      path = e["path"]
      parts = path.split("/", -1)
      if path.start_with?("/") || parts.include?("..") || path.include?("\\")
        raise Fatal, "refusing unsafe path from server: #{path.inspect}"
      end
      # Control characters would be sent to the terminal as escape codes.
      if path.match?(/[\u0000-\u001f\u007f-\u009f]/)
        raise Fatal, "refusing path with control characters: #{path.inspect}"
      end
      if parts.any? { |p| dotgit?(p) }
        # A file under .git could plant git config or hooks that run
        # commands as soon as git is used in the folder.
        raise Fatal, "refusing repo with a file inside a .git folder: #{path.inspect}"
      end

      lfs = e["lfs"]
      files << if lfs
                 RepoFile.new(path, lfs["size"] || e["size"], "sha256", lfs["oid"])
               else
                 RepoFile.new(path, e["size"], "git-sha1", e["oid"])
               end
    end
    m = /<([^>]+)>;\s*rel="next"/.match(res["link"] || "")
    url = m ? URI.join(url, m[1]).to_s : nil
  end
  [commit, files]
end

# ---------------------------------------------------------------- local state

def mtime_ns(stat)
  stat.mtime.to_i * 1_000_000_000 + stat.mtime.nsec
end

# Records verified files so restarts need not re-hash them.
class Manifest
  def initialize(path)
    @path = path
    @lock = Mutex.new
    @data = begin
      parsed = JSON.parse(File.read(path))
      parsed.is_a?(Hash) ? parsed : {}
    rescue SystemCallError, JSON::ParserError
      {}
    end
  end

  def matches?(file, stat)
    @data[file.path] == [file.digest, stat.size, mtime_ns(stat)]
  end

  def record(file, stat)
    @lock.synchronize do
      @data[file.path] = [file.digest, stat.size, mtime_ns(stat)]
      tmp = "#{@path}.tmp"
      File.write(tmp, JSON.generate(@data, indent: "", object_nl: "\n", array_nl: "\n"))
      File.rename(tmp, @path)
    end
  end
end

# Network bytes over time, for a current and a 5-minute average speed.
class Rate
  attr_accessor :fetched

  def initialize
    @fetched = 0
    @samples = [[mono, 0]] # [time, fetched so far]
  end

  def sample(now)
    return unless now - @samples[-1][0] >= 0.5

    @samples << [now, @fetched]
    @samples.shift while @samples.length > 2 && @samples[1][0] <= now - AVG_WINDOW
  end

  # Bytes/s over the last `window` seconds (or since the start, if shorter).
  def speed(now, window)
    t0, n0 = @samples[0]
    @samples.reverse_each do |t, n|
      t0 = t
      n0 = n
      break if t <= now - window
    end
    (@fetched - n0) / [now - t0, 1e-6].max
  end
end

# Live display: one bar per file being downloaded plus a total bar.
class Progress
  attr_reader :total

  def initialize(total_bytes, total_files)
    @total = total_bytes
    @nfiles = total_files
    @done = 0
    @files_done = 0
    @rate = Rate.new
    @active = {}    # path -> [bytes on disk, size, Rate]
    @drawn = 0      # lines of the live region currently on screen
    @lock = Mutex.new
    @tty = $stderr.tty?
    @last = mono
  end

  def start(file)
    @lock.synchronize { @active[file.path] = [0, file.size, Rate.new] }
  end

  def add(count, network: false, path: nil)
    @lock.synchronize do
      @done += count
      @rate.fetched += count if network
      entry = @active[path]
      if entry
        entry[0] += count
        entry[2].fetched += count if network
      end
    end
  end

  def finish(file, result)
    @lock.synchronize do
      was_active = @active.delete(file.path)
      @files_done += 1 if %i[ok skipped].include?(result)
      println("✓ #{file.path}  (#{human(file.size)}, verified)") if was_active && result == :ok
    end
  end

  def log(msg)
    @lock.synchronize { println(msg) }
  end

  def show(force: false)
    now = mono
    return if !force && now - @last < (@tty ? 0.2 : 30)

    @last = now
    @lock.synchronize do
      @rate.sample(now)
      @active.each_value { |entry| entry[2].sample(now) }
      if @tty
        render
      else
        pct = @total.positive? ? 100.0 * @done / @total : 100.0
        $stderr.puts format("%d/%d files  %s/%s  %5.1f%%  %s/s  5m avg %s/s",
                            @files_done, @nfiles, human(@done), human(@total), pct,
                            human(@rate.speed(now, NOW_WINDOW)), human(@rate.speed(now, AVG_WINDOW)))
        $stderr.flush
      end
    end
  end

  private

  # The methods below expect @lock to be held.

  def println(msg)
    render(msg)
  end

  # Redraw the live region, optionally printing a permanent line above it.
  #
  # The new frame overwrites the old one in place and goes out in a single
  # write. Erasing the region first would leave it blank for a moment, which
  # shows up as flashing.
  def render(msg = nil)
    unless @tty
      unless msg.nil?
        $stderr.write("#{msg}\n")
        $stderr.flush
      end
      return
    end
    cols, rows = terminal_size
    now = mono
    lines = @active.first([rows - 2, 1].max).map do |path, (done, size, rate)|
      bar_line(path, done, size, rate, now, cols - 1)
    end
    lines << bar_line("TOTAL #{@files_done}/#{@nfiles} files", @done, @total, @rate, now, cols - 1)
    out = +""
    # back to the first line of the old frame
    out << "\r" << (@drawn > 1 ? "\e[#{@drawn - 1}A" : "") if @drawn.positive?
    out << msg << "\e[K\n" unless msg.nil?
    # \e[K clears the rest of each line, \e[J whatever is left below.
    out << lines.join("\e[K\n") << "\e[K\e[J"
    $stderr.write(out)
    $stderr.flush
    @drawn = lines.length
  end
end

# -> [columns, rows]
def terminal_size
  cols = ENV["COLUMNS"].to_i
  rows = ENV["LINES"].to_i
  if cols <= 0 || rows <= 0
    begin
      r, c = $stderr.winsize
      cols = c if cols <= 0
      rows = r if rows <= 0
    rescue StandardError
      nil
    end
  end
  [cols.positive? ? cols : 80, rows.positive? ? rows : 24]
end

def bar_line(label, done, size, rate, now, width)
  cur = rate.speed(now, NOW_WINDOW)
  avg = rate.speed(now, AVG_WINDOW)
  frac = size.positive? ? [done.to_f / size, 1.0].min : 1.0
  eta = if done >= size
          "0:00:00"
        elsif avg.positive? # the average is steadier than the current speed
          secs = ((size - done) / avg).to_i
          format("%d:%02d:%02d", secs / 3600, secs % 3600 / 60, secs % 60)
        else
          "-:--:--"
        end
  tail = format(" %5.1f%%  %9s/%-9s  %9s/s  5m avg %9s/s  ETA %s",
                frac * 100, human(done), human(size), human(cur), human(avg), eta)
  name_w = [12, [40, width / 5].min].max
  bar_w = width - name_w - tail.length - 3
  name_w = [8, width - tail.length].max if bar_w < 5 # narrow terminal: drop the bar, keep the numbers
  label = "…#{label[-(name_w - 1)..]}" if label.length > name_w
  return (label.ljust(name_w) + tail)[0, width] if bar_w < 5

  fill = (frac * bar_w).to_i
  "#{label.ljust(name_w)} [#{"█" * fill}#{"░" * (bar_w - fill)}]#{tail}"
end

def human(num)
  num = num.to_f
  %w[B KiB MiB GiB].each do |unit|
    return unit == "B" ? format("%.0f B", num) : format("%.1f %s", num, unit) if num < 1024

    num /= 1024
  end
  format("%.1f TiB", num)
end

def hash_file(path, file)
  hasher = file.new_hasher
  File.open(path, "rb") do |fh|
    while (chunk = fh.read(CHUNK))
      break if STOP.set?

      hasher.update(chunk)
    end
  end
  hasher
end

# ---------------------------------------------------------------- download

# Absolute path with symlinks resolved; the tail of the path need not exist.
def resolve_path(path)
  path = File.expand_path(path)
  rest = []
  until File.exist?(path) || File.symlink?(path)
    parent = File.dirname(path)
    break if parent == path

    rest.unshift(File.basename(path))
    path = parent
  end
  File.join(File.realpath(path), *rest)
end

# Where the file belongs on disk; refuses anything that would land outside root.
def dest_path(root, file)
  dest = File.join(root, file.path)
  # Belt and braces on top of the listing check: also catches symlinks that
  # lead out of the folder and Windows drive-style names like "C:x".
  resolved = begin
    resolve_path(dest)
  rescue SystemCallError
    nil # e.g. a dangling symlink in the way
  end
  unless resolved && (resolved == root || resolved.start_with?("#{root}/"))
    raise Fatal, "#{file.path}: would be written outside #{root}, skipping"
  end

  dest
end

# True if dest is a verified copy of the file. Hashes it unless the manifest vouches for it.
def already_complete?(file, dest, manifest, verify)
  return false unless File.file?(dest)

  stat = File.stat(dest)
  return false if stat.size != file.size
  return true if !verify && manifest.matches?(file, stat)

  if hash_file(dest, file).hexdigest == file.digest && !STOP.set?
    manifest.record(file, stat)
    return true
  end
  false
end

# Fetch one file. Returns :ok, :skipped or :interrupted; raises Fatal.
def download_file(file, url, root, token, manifest, progress, verify, retries)
  dest = dest_path(root, file)
  part = "#{dest}.#{file.digest[0, 12]}.part"
  FileUtils.mkdir_p(File.dirname(dest))

  # Partials belonging to an older version of this file are useless now.
  prefix = "#{File.basename(dest)}."
  Dir.children(File.dirname(dest)).each do |name|
    next unless name.start_with?(prefix) && name.end_with?(".part") && name.length >= prefix.length + 5

    stale = File.join(File.dirname(dest), name)
    File.unlink(stale) if stale != part
  end

  if already_complete?(file, dest, manifest, verify)
    progress.add(file.size)
    return :skipped
  end
  return :interrupted if STOP.set?

  if File.exist?(dest)
    progress.log("#{file.path}: existing file is corrupt or outdated, re-downloading")
    File.unlink(dest)
  end

  progress.start(file)
  hasher = nil
  hashed = -1     # hasher state covers the first `hashed` bytes of part
  counted = 0     # bytes of this file already shown in progress
  failures = 0
  while true # rubocop:disable Style/InfiniteLoop -- the body shares locals with the block below
    return :interrupted if STOP.set?

    have = File.exist?(part) ? File.size(part) : 0
    if have > file.size # cannot be a prefix of the right file
      File.unlink(part)
      have = 0
    end
    if hashed != have # first pass / after an error: re-sync hash with disk
      hasher = have.positive? ? hash_file(part, file) : file.new_hasher
      hashed = have
      return :interrupted if STOP.set?
    end
    progress.add(have - counted, path: file.path)
    counted = have
    before = have

    error = nil
    begin
      if file.size.zero?
        FileUtils.touch(part)
      elsif have < file.size
        http_get(url, token, have) do |res|
          if have.positive? && res.code != "206" # server ignored Range: start over
            progress.add(-counted, path: file.path)
            have = counted = hashed = 0
            hasher = file.new_hasher
          end
          File.open(part, have.positive? ? "ab" : "wb") do |out|
            res.read_body do |chunk|
              return :interrupted if STOP.set?

              out.write(chunk)
              hasher.update(chunk)
              hashed += chunk.bytesize
              counted += chunk.bytesize
              progress.add(chunk.bytesize, network: true, path: file.path)
            end
            out.flush
            out.fsync
          end
        end
      end
    rescue HTTPError => e
      if [401, 403, 404].include?(e.code)
        raise Fatal, "#{file.path}: HTTP #{e.code} (gated/private repo? set HF_TOKEN)"
      end

      FileUtils.rm_f(part) if e.code == 416
      error = e.message
    rescue Errno::ENOSPC, Errno::EDQUOT, Errno::EROFS => e
      raise Fatal, "#{file.path}: #{e.message}"
    rescue *NETWORK_ERRORS => e
      error = e.message
    end

    size = File.exist?(part) ? File.size(part) : 0
    if size == file.size && hashed == size
      if hasher.hexdigest == file.digest
        File.rename(part, dest)
        manifest.record(file, File.stat(dest))
        return :ok
      end
      error = "checksum mismatch, discarding the downloaded data"
      File.unlink(part)
    elsif error.nil? && size >= file.size
      next # sizes disagree with our bookkeeping: loop re-syncs and re-checks
    end

    failures = size > before ? 0 : failures + 1
    raise Fatal, "#{file.path}: giving up after #{retries} retries (#{error})" if failures > retries

    next if error.nil?

    delay = [60, 2**failures].min
    progress.log("#{file.path}: #{error} - retrying in #{delay}s")
    STOP.wait(delay)
  end
end

# ---------------------------------------------------------------- git

GitResult = Struct.new(:stdout, :ok)

def which(program)
  ENV.fetch("PATH", "").split(File::PATH_SEPARATOR).each do |dir|
    candidate = File.join(dir.empty? ? "." : dir, program)
    return candidate if File.file?(candidate) && File.executable?(candidate)
  end
  nil
end

# Turn the fully downloaded folder into a git clone checked out at `commit`.
#
# Only git's own objects are fetched (history, small files, LFS pointers).
# Every file on disk has just been verified against that commit, so the
# index is filled in without making git re-read the large files.
def git_convert(root, kind, repo, rev, commit, token)
  git = which("git")
  raise Fatal, "git is not installed" unless git
  raise Fatal, "unexpected commit id from server: #{commit.inspect}" unless commit.match?(/\A[0-9a-f]{40,64}\z/)

  url = "#{ENDPOINT}/#{kind == 'models' ? '' : "#{kind}/"}#{repo}"
  say = lambda do |msg|
    $stderr.puts msg
    $stderr.flush
  end

  # Copy of environment `base` that passes extra `git -c key=value` settings.
  with_config = lambda do |base, pairs|
    n = base.fetch("GIT_CONFIG_COUNT", "0").to_i
    out = base.merge("GIT_CONFIG_COUNT" => (n + pairs.length).to_s)
    pairs.each_with_index do |(key, value), i|
      out["GIT_CONFIG_KEY_#{n + i}"] = key
      out["GIT_CONFIG_VALUE_#{n + i}"] = value
    end
    out
  end

  # No hooks and no fsmonitor command for the git calls made here, whatever a
  # .git folder that is already present may say.
  env = with_config.call(ENV.to_h.merge("GIT_TERMINAL_PROMPT" => "0", "GIT_LFS_SKIP_SMUDGE" => "1"),
                         [["core.hooksPath", File::NULL], ["core.fsmonitor", "false"]])
  auth_env = env
  if token # for the fetch only, via the environment so it is not visible in `ps`
    basic = ["hf_user:#{token}"].pack("m0")
    auth_env = with_config.call(env, [["http.#{ENDPOINT}/.extraHeader", "Authorization: Basic #{basic}"]])
  end

  run = lambda do |*cmd, check: true, env_for: env, config: []|
    full = [git] + config.flat_map { |kv| ["-c", kv] } + cmd
    out, err, status = Open3.capture3(env_for, *full, chdir: root)
    if check && !status.success?
      raise Fatal, "git #{cmd[0]} failed: #{(err.empty? ? out : err).strip}"
    end

    GitResult.new(out, status.success?)
  end

  say.call("turning #{File.basename(root)} into a git clone of #{url} ...")
  # checked by path: a parent folder may be a repo too
  run.call("init", "-q") unless File.exist?(File.join(root, ".git"))
  origin = run.call("remote", "get-url", "origin", check: false)
  if !origin.ok
    run.call("remote", "add", "origin", url)
  elsif origin.stdout.strip.sub(%r{/+\z}, "").delete_suffix(".git") != url
    raise Fatal, "#{root} is already a git repo with another origin " \
                 "(#{origin.stdout.strip}); leaving it untouched"
  end

  run.call("fetch", "-q", "origin", env_for: auth_env)
  unless run.call("cat-file", "-e", "#{commit}^{commit}", check: false).ok
    run.call("fetch", "-q", "origin", commit, env_for: auth_env) # commit not on any branch/tag
  end

  head = run.call("rev-parse", "-q", "--verify", "HEAD", check: false).stdout.strip
  if head.empty?
    # New repo. If the page was a branch, check that branch out and make it
    # track origin; for a tag or commit id, leave a detached HEAD.
    if run.call("rev-parse", "-q", "--verify", "refs/remotes/origin/#{rev}", check: false).ok
      run.call("symbolic-ref", "HEAD", "refs/heads/#{rev}")
      run.call("update-ref", "HEAD", commit)
      run.call("config", "branch.#{rev}.remote", "origin")
      run.call("config", "branch.#{rev}.merge", "refs/heads/#{rev}")
    else
      run.call("update-ref", "--no-deref", "HEAD", commit)
    end
  elsif head != commit
    # The folder was a clone already and the repo has moved on since.
    unless run.call("merge-base", "--is-ancestor", head, commit, check: false).ok
      raise Fatal, "the git repo in #{root} has commits that are not in #{commit[0, 12]}; " \
                   "leaving its git state untouched"
    end
    run.call("update-ref", "HEAD", commit)
  end

  # origin/HEAD, as clone does
  run.call("remote", "set-head", "origin", "--auto", check: false, env_for: auth_env)

  # Point the index at the commit without touching the files on disk.
  run.call("read-tree", commit)

  exclude = File.join(root, ".git", "info", "exclude")
  FileUtils.mkdir_p(File.dirname(exclude))
  have = File.exist?(exclude) ? File.read(exclude) : ""
  missing = [MANIFEST, "#{MANIFEST}.tmp", "*.part"] - have.split("\n")
  unless missing.empty?
    separator = have.end_with?("\n") || have.empty? ? "" : "\n"
    File.write(exclude, "#{have}#{separator}#{missing.join("\n")}\n")
  end

  # Let git record each file as unchanged. Normally it would push every LFS
  # file through `git lfs clean`, i.e. read and hash all of it again (and
  # git-lfs would keep a second copy under .git/lfs). The checksums were
  # verified moments ago, so for this one command the LFS filter is replaced
  # by one that answers with the pointer stored in the commit. Small non-LFS
  # files are still hashed by git itself. The user's global/system git config
  # is hidden for this command, because git-lfs registers a filter there that
  # would take precedence.
  sleep(1.1) # files as new as the index would be re-checked by git later
  bare_env = env.merge("GIT_CONFIG_GLOBAL" => File::NULL, "GIT_CONFIG_SYSTEM" => File::NULL,
                       "GIT_CONFIG_NOSYSTEM" => "1")
  run.call("update-index", "-q", "--refresh", check: false, env_for: bare_env,
           config: ["filter.lfs.clean=git cat-file blob #{commit}:%f", "filter.lfs.required=false"])

  status = run.call("status", "--porcelain").stdout.split("\n")
  branch = run.call("symbolic-ref", "-q", "--short", "HEAD", check: false).stdout.strip
  say.call("git clone ready: #{branch.empty? ? 'detached HEAD' : "branch #{branch}"} at #{commit[0, 12]}")
  unless status.empty?
    say.call("note: git reports #{status.length} difference(s) from that commit:")
    status.first(10).each { |line| say.call("  #{line}") }
  end
  return if run.call("lfs", "version", check: false).ok

  say.call("note: git-lfs is not installed. The folder is a valid clone, but install git-lfs " \
           "before running\n      git pull/checkout in it, or large files will be replaced " \
           "by small pointer files.")
end

# ---------------------------------------------------------------- main

def parse_args(argv)
  opts = { dir: ".", jobs: 4, retries: 10, verify: false, nogit: false, togit: false }
  parser = OptionParser.new do |o|
    o.banner = "usage: hf_download.rb [options] url\n\n" \
               "Download every file of a Hugging Face repo page, resumably and verified.\n\n" \
               "    url    Hugging Face page URL, e.g. https://huggingface.co/org/name/tree/main\n\n" \
               "options:"
    o.on("-d", "--dir DIR", "where to create the repo's folder (default: .)") { |v| opts[:dir] = v }
    o.on("-j", "--jobs JOBS", Integer, "parallel downloads (default: 4)") { |v| opts[:jobs] = v }
    o.on("--retries RETRIES", Integer,
         "consecutive attempts without any progress before", "giving up on a file") { |v| opts[:retries] = v }
    o.on("--verify", "re-hash files that are already complete instead of",
         "trusting the manifest") { opts[:verify] = true }
    o.on("--nogit", "only download the files; don't turn the folder into",
         "a git clone") { opts[:nogit] = true }
    o.on("--togit", "download nothing; turn an existing, complete",
         "download into a git clone") { opts[:togit] = true }
    o.on("-h", "--help", "show this help message and exit") do
      puts o
      exit 0
    end
  end
  usage_error = lambda do |msg|
    warn parser.banner.lines.first
    warn "hf_download.rb: error: #{msg}"
    exit 2
  end
  begin
    rest = parser.parse(argv)
  rescue OptionParser::ParseError => e
    usage_error.call(e.message)
  end
  usage_error.call("argument --togit: not allowed with argument --nogit") if opts[:nogit] && opts[:togit]
  usage_error.call("the following arguments are required: url") if rest.empty?
  usage_error.call("unrecognized arguments: #{rest[1..].join(' ')}") if rest.length > 1
  opts[:url] = rest[0]
  opts
end

def main(argv)
  args = parse_args(argv)

  token = find_token
  kind, repo, rev, sub = parse_url(args[:url])
  commit, files = list_files(kind, repo, rev, sub, token)
  raise Fatal, "no files found on that page" if files.empty?

  folder = repo.gsub("/", "__") + (rev == "main" ? "" : "@#{rev.gsub('/', '_')}")
  root = resolve_path(File.join(args[:dir], folder))
  if args[:togit]
    raise Fatal, "a subfolder download cannot become a git clone; use the repo's main page URL" unless sub.empty?
    raise Fatal, "#{root} does not exist, so there is nothing to convert" unless File.directory?(root)

    manifest = Manifest.new(File.join(root, MANIFEST))
    warn "checking #{files.length} files in #{root} ..."
    bad = files.reject { |f| already_complete?(f, dest_path(root, f), manifest, args[:verify]) }.map(&:path)
    unless bad.empty?
      raise Fatal, "#{bad.length} of #{files.length} files are missing, incomplete or outdated " \
                   "(e.g. #{bad[0]}); run without --togit to finish the download first"
    end
    git_convert(root, kind, repo, rev, commit, token)
    return 0
  end

  FileUtils.mkdir_p(root)
  manifest = Manifest.new(File.join(root, MANIFEST))
  progress = Progress.new(files.sum(&:size), files.length)
  warn "#{repo} @ #{commit[0, 12]}: #{files.length} files, #{human(progress.total)} -> #{root}"

  prefix = kind == "models" ? "" : "#{kind}/"
  base = "#{ENDPOINT}/#{prefix}#{quote(repo)}/resolve/#{commit}/"
  failed = []
  failed_lock = Mutex.new
  interrupted = false

  job = lambda do |f|
    result = :failed
    begin
      result = download_file(f, base + quote(f.path), root, token, manifest, progress,
                             args[:verify], args[:retries])
    rescue Fatal => e
      failed_lock.synchronize { failed << e.message }
      progress.log("FAILED #{e.message}")
    rescue StandardError => e # keep the other downloads going
      failed_lock.synchronize { failed << "#{f.path}: #{e.inspect}" }
      progress.log("FAILED #{f.path}: #{e.inspect}")
    ensure
      progress.finish(f, result)
    end
  end

  # Small files first: the folder becomes usable/inspectable quickly.
  queue = Queue.new
  files.each_with_index.sort_by { |f, i| [f.size, i] }.each { |f, _| queue << f }
  workers = Array.new([args[:jobs], 1].max) do
    Thread.new do
      until STOP.set?
        f = begin
          queue.pop(true)
        rescue ThreadError
          break
        end
        job.call(f)
      end
    end
  end
  begin
    while workers.any?(&:alive?)
      progress.show
      sleep(0.2)
    end
  rescue Interrupt
    interrupted = true
    STOP.set
    progress.log("interrupted - partial files are kept, run the same command to resume")
  end
  workers.each(&:join)
  progress.show(force: true)
  warn ""

  return 130 if interrupted

  unless failed.empty?
    warn "#{failed.length} file(s) failed; run the same command again to retry:"
    failed.each { |msg| warn "  #{msg}" }
    return 1
  end
  warn "done: all #{files.length} files downloaded and verified in #{root}"
  return 0 if args[:nogit]

  unless sub.empty?
    warn "note: only a subfolder was downloaded, so it was not turned into a git clone"
    return 0
  end
  begin
    git_convert(root, kind, repo, rev, commit, token)
  rescue Fatal => e
    warn "error: the files are downloaded and verified, but making the folder a git clone " \
         "failed:\n  #{e.message}\nRun again with --togit to retry, or with --nogit to skip this step."
    return 1
  end
  0
end

if $PROGRAM_NAME == __FILE__
  begin
    exit(main(ARGV))
  rescue Fatal => e
    abort("error: #{e.message}")
  rescue Interrupt
    exit(130)
  end
end
