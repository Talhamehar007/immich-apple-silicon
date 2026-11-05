import CryptoKit
import Foundation

// Checks a downloaded model file against the digest Hugging Face publishes for
// the same revision, so a corrupted or substituted body is refused instead of
// handed to onnxruntime or mlx.
//
// Why this exists: the only guard before it was Content-Length. That catches a
// truncated transfer, which is the common failure, and nothing else. A file of
// the right length with the wrong bytes was accepted, cached, and loaded
// forever after — the size>0 "already downloaded" check means it is never
// fetched again. Weights are executable input to two C/C++ runtimes, and the
// largest of them are several GB from a CDN.
//
// The listing endpoint is the independent statement of what each file should
// contain. Two forms appear there:
//
//   lfs.oid    SHA-256 of the bytes, for anything stored in Git LFS (weights)
//   oid        the git blob id — SHA-1 over "blob <size>\0" + the bytes
//
// Both are checked with the same streaming pass, so a 4.5 GB file is never
// held in memory a second time.
//
// What this does NOT do: pin a digest in the source tree. The expectation is
// whatever Hugging Face says at download time, over TLS. That detects
// corruption and a CDN serving the wrong object; it does not defend against a
// compromised upstream account. Pinning `immich-app` zoo digests in the tree
// would be the next step, and is deliberately not pretended here.
//
// Asking the API is best effort. A machine that cannot reach it still gets its
// models, with one warning per repository, because a listing endpoint having a
// bad afternoon must not stop a library from being processed.
enum ModelIntegrity {
    enum Expectation {
        case sha256(String)
        case gitBlobSha1(String, size: Int)
    }

    private static let lock = NSLock()
    nonisolated(unsafe) private static var manifests: [String: [String: Expectation]] = [:]
    nonisolated(unsafe) private static var unavailable: Set<String> = []
    nonisolated(unsafe) private static var announced: Set<String> = []

    /// nil when the file matches, or when there is nothing to compare it to.
    /// Otherwise a sentence explaining what is wrong with it.
    static func mismatch(fileAt url: URL, path: String, repo: String) -> String? {
        guard let table = manifest(repo: repo), let expected = table[path] else { return nil }
        let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        if size == 0 {
            return "\(path) is empty"
        }
        let algorithm: String
        let want: String
        switch expected {
        case .sha256(let hex):
            algorithm = "SHA-256"
            want = hex
        case .gitBlobSha1(let hex, let expectedSize):
            algorithm = "SHA-1"
            want = hex
            if size != expectedSize {
                return "\(path) is \(size) bytes, expected \(expectedSize)"
            }
        }
        guard let got = digest(fileAt: url, gitBlob: algorithm == "SHA-1") else {
            return "could not read \(path) back to check it"
        }
        if got != want {
            return "\(path) failed its \(algorithm) check "
                + "(got \(got.prefix(12))…, expected \(want.prefix(12))…)"
        }
        return nil
    }

    // MARK: - Manifest

    /// path -> expectation for one repository, or nil when the API could not be
    /// asked. Cached for the life of the process; a model is fetched once.
    private static func manifest(repo: String) -> [String: Expectation]? {
        lock.lock()
        if let known = manifests[repo] {
            lock.unlock()
            return known
        }
        if unavailable.contains(repo) {
            lock.unlock()
            return nil
        }
        lock.unlock()

        let fetched = fetchManifest(repo: repo)

        lock.lock()
        if let fetched {
            manifests[repo] = fetched
            if announced.insert(repo).inserted {
                print("[native-ml] \(repo): \(fetched.count) files carry a published digest")
            }
        } else if announced.insert(repo).inserted {
            print("[native-ml] \(repo): could not read Hugging Face's file listing; "
                + "downloads are checked for length only")
        }
        if fetched == nil { unavailable.insert(repo) }
        lock.unlock()
        return fetched
    }

    private static func fetchManifest(repo: String) -> [String: Expectation]? {
        guard let url = URL(
            string: "https://huggingface.co/api/models/\(repo)/tree/main?recursive=true")
        else { return nil }
        var req = URLRequest(url: url)
        req.timeoutInterval = 30
        // Same additive token SigLIPNative uses: raises the rate limit, and is
        // needed at all for a gated repository.
        if let token = ProcessInfo.processInfo.environment["HF_TOKEN"], !token.isEmpty {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        let sem = DispatchSemaphore(value: 0)
        var payload: Data?
        URLSession(configuration: .ephemeral).dataTask(with: req) { data, _, _ in
            payload = data
            sem.signal()
        }.resume()
        sem.wait()

        guard let data = payload,
              let rows = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]]
        else { return nil }

        var out: [String: Expectation] = [:]
        for row in rows {
            guard let path = row["path"] as? String else { continue }
            if let lfs = row["lfs"] as? [String: Any], let oid = lfs["oid"] as? String {
                out[path] = .sha256(oid.lowercased())
            } else if let oid = row["oid"] as? String,
                      let size = (row["size"] as? NSNumber)?.intValue {
                out[path] = .gitBlobSha1(oid.lowercased(), size: size)
            }
        }
        return out.isEmpty ? nil : out
    }

    // MARK: - Hashing

    /// 64 lowercase hex characters, or nil when the file cannot be read.
    /// *gitBlob* hashes the git object id instead of the raw bytes.
    private static func digest(fileAt url: URL, gitBlob: Bool) -> String? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }

        if gitBlob {
            guard let size = (try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) else {
                return nil
            }
            var sha1 = Insecure.SHA1()
            sha1.update(data: Data("blob \(size)\0".utf8))
            guard readAll(handle, { sha1.update(data: $0) }) else { return nil }
            return sha1.finalize().map { String(format: "%02x", $0) }.joined()
        }
        var sha256 = SHA256()
        guard readAll(handle, { sha256.update(data: $0) }) else { return nil }
        return sha256.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Feeds every 4 MiB chunk to *body*. False when the file stopped reading
    /// early, so a half-read file is never reported as a mismatch (which would
    /// delete a good download over a transient I/O error).
    private static func readAll(_ handle: FileHandle, _ body: (Data) -> Void) -> Bool {
        while true {
            let chunk: Data?
            do {
                chunk = try handle.read(upToCount: 4 << 20)
            } catch {
                return false
            }
            guard let chunk, !chunk.isEmpty else { return true }
            body(chunk)
        }
    }
}
