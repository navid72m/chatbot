const fs = require("fs");
const path = require("path");
const http = require("http");
const https = require("https");
const crypto = require("crypto");
const { URL } = require("url");

function request(url, opts = {}) {
  const u = new URL(url);
  const lib = u.protocol === "https:" ? https : http;

  const req = lib.request(
    {
      method: opts.method || "GET",
      hostname: u.hostname,
      path: u.pathname + u.search,
      headers: opts.headers || {},
      timeout: opts.timeoutMs ?? 30000,
    },
    opts.onResponse
  );

  req.on("timeout", () => {
    req.destroy(new Error("Request timed out"));
  });

  return req;
}

async function head(url, headers = {}) {
  // Follow redirects and return { finalUrl, headers }
  return new Promise((resolve, reject) => {
    const req = request(url, {
      method: "HEAD",
      headers,
      onResponse: (res) => {
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          // Some redirects return relative URLs
          const redirected = new URL(res.headers.location, url).toString();
          res.resume();
          return resolve(head(redirected, headers));
        }
        if (res.statusCode < 200 || res.statusCode >= 400) {
          res.resume();
          return reject(new Error(`HEAD failed: ${res.statusCode}`));
        }
        res.resume();
        resolve({ finalUrl: url, headers: res.headers, statusCode: res.statusCode });
      },
    });
    req.on("error", reject);
    req.end();
  });
}

function sha256File(filePath) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    const stream = fs.createReadStream(filePath);
    stream.on("error", reject);
    stream.on("data", (d) => hash.update(d));
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}

/**
 * ensureModelDownloaded
 * - Downloads to destPath + ".part" with resume if server supports ranges
 * - Optional SHA256 verification (recommended for GGUF)
 */
async function ensureModelDownloaded({ url, destPath, onProgress, hfToken, sha256 }) {
  fs.mkdirSync(path.dirname(destPath), { recursive: true });

  const authHeaders = {};
  if (hfToken) authHeaders["Authorization"] = `Bearer ${hfToken}`;

  // If already downloaded, verify (if sha256 provided)
  if (fs.existsSync(destPath)) {
    if (sha256) {
      const got = await sha256File(destPath);
      if (got.toLowerCase() !== sha256.toLowerCase()) {
        // Corrupt/old file; remove and re-download
        fs.unlinkSync(destPath);
      } else {
        return destPath;
      }
    } else {
      return destPath;
    }
  }

  // Resolve final URL + get size + range support
  const h = await head(url, authHeaders);
  const finalUrl = h.finalUrl;

  const total = parseInt(h.headers["content-length"] || "0", 10);
  const acceptRanges = (h.headers["accept-ranges"] || "").toLowerCase().includes("bytes");

  const partPath = destPath + ".part";
  let existing = 0;
  if (fs.existsSync(partPath)) existing = fs.statSync(partPath).size;

  // If server doesn't support ranges, restart from scratch
  if (!acceptRanges) existing = 0;

  await new Promise((resolve, reject) => {
    const headers = { ...authHeaders };
    if (existing > 0) headers["Range"] = `bytes=${existing}-`;

    const req = request(finalUrl, {
      headers,
      timeoutMs: 30000,
      onResponse: (res) => {
        // Follow redirects for GET too
        if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
          const redirected = new URL(res.headers.location, finalUrl).toString();
          res.resume();
          req.destroy();
          ensureModelDownloaded({ url: redirected, destPath, onProgress, hfToken, sha256 })
            .then(resolve)
            .catch(reject);
          return;
        }

        // 200 = full download, 206 = resumed partial
        if (![200, 206].includes(res.statusCode)) {
          res.resume();
          return reject(new Error(`Download failed: ${res.statusCode}`));
        }

        // If we asked for Range but server ignored it, restart clean
        if (existing > 0 && res.statusCode === 200) {
          existing = 0;
        }

        const file = fs.createWriteStream(partPath, { flags: existing > 0 ? "a" : "w" });

        let downloaded = existing;

        // If we got a ranged response, content-length is "remaining bytes"
        const contentLen = parseInt(res.headers["content-length"] || "0", 10);
        const totalFromHeaders =
          total || (res.statusCode === 206 ? existing + contentLen : contentLen);

        res.on("data", (chunk) => {
          downloaded += chunk.length;
          if (onProgress && totalFromHeaders > 0) {
            onProgress({
              downloaded,
              total: totalFromHeaders,
              percent: downloaded / totalFromHeaders,
            });
          }
        });

        res.pipe(file);

        file.on("finish", () => file.close(resolve));
        file.on("error", (e) => {
          try { file.close(); } catch {}
          reject(e);
        });
      },
    });

    req.on("error", reject);
    req.end();
  });

  // Verify checksum if provided
  if (sha256) {
    const got = await sha256File(partPath);
    if (got.toLowerCase() !== sha256.toLowerCase()) {
      try { fs.unlinkSync(partPath); } catch {}
      throw new Error("Model checksum mismatch (download corrupted)");
    }
  }

  fs.renameSync(partPath, destPath);
  return destPath;
}

module.exports = { ensureModelDownloaded };
