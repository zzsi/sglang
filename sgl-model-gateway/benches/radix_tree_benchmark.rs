//! Benchmarks for the radix tree implementations used in cache-aware routing.
//!
//! This benchmark compares both implementations:
//! - StringTree: Character-based tree for HTTP router (text input)
//! - TokenTree: Token-based tree for gRPC router (pre-tokenized input)
//!
//! Run with: cargo bench --bench radix_tree_benchmark
//!
//! For quick validation: cargo bench --bench radix_tree_benchmark -- benchmark_summary --exact

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread,
    time::Instant,
};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::{
    distr::{Alphanumeric, SampleString},
    rng as thread_rng, Rng,
};
// Import both old and new tree implementations
use sgl_model_gateway::policies::tree::Tree as OldTree;
use sgl_model_gateway::radix_tree::{StringTree, TokenTree};

// Global results storage for summary
lazy_static::lazy_static! {
    static ref BENCHMARK_RESULTS: Mutex<BTreeMap<String, String>> = Mutex::new(BTreeMap::new());
}

fn add_result(category: &str, result: String) {
    let mut results = BENCHMARK_RESULTS.lock().unwrap();
    let index = results.len();
    let key = format!("{:03}_{}", index, category);
    eprintln!("[BENCH_RESULT] {} | {}", category, result);
    results.insert(key, result);
}

/// Simulated HTTP/gRPC endpoints representing worker nodes
const ENDPOINT_TENANTS: [&str; 10] = [
    "http://worker-0.sglang.svc.cluster.local:8000",
    "http://worker-1.sglang.svc.cluster.local:8000",
    "http://worker-2.sglang.svc.cluster.local:8000",
    "http://worker-3.sglang.svc.cluster.local:8000",
    "http://worker-4.sglang.svc.cluster.local:8000",
    "grpc://worker-5.sglang.svc.cluster.local:50051",
    "grpc://worker-6.sglang.svc.cluster.local:50051",
    "grpc://worker-7.sglang.svc.cluster.local:50051",
    "http://10.0.0.100:8000",
    "http://10.0.0.101:8000",
];

/// Common conversation prefixes that create shared tree paths
const CONVERSATION_PREFIXES: [&str; 6] = [
    "<|system|>\nYou are a helpful assistant.\n<|user|>\n",
    "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n",
    "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n",
    "Human: ",
    "User: ",
    "### Instruction:\n",
];

/// Common token prefixes that create shared tree paths (mirrors CONVERSATION_PREFIXES)
/// These simulate tokenized versions of common system prompts
const TOKEN_PREFIXES: [[TokenId; 20]; 6] = [
    // Simulates "<|system|>\nYou are a helpful assistant.\n<|user|>\n"
    [
        100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
        1700, 1800, 1900, 2000,
    ],
    // Simulates "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n<|im_start|>user\n"
    [
        101, 201, 301, 401, 501, 601, 701, 801, 901, 1001, 1101, 1201, 1301, 1401, 1501, 1601,
        1701, 1801, 1901, 2001,
    ],
    // Simulates "[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
    [
        102, 202, 302, 402, 502, 602, 702, 802, 902, 1002, 1102, 1202, 1302, 1402, 1502, 1602,
        1702, 1802, 1902, 2002,
    ],
    // Simulates "Human: "
    [
        103, 203, 303, 403, 503, 603, 703, 803, 903, 1003, 1103, 1203, 1303, 1403, 1503, 1603,
        1703, 1803, 1903, 2003,
    ],
    // Simulates "User: "
    [
        104, 204, 304, 404, 504, 604, 704, 804, 904, 1004, 1104, 1204, 1304, 1404, 1504, 1604,
        1704, 1804, 1904, 2004,
    ],
    // Simulates "### Instruction:\n"
    [
        105, 205, 305, 405, 505, 605, 705, 805, 905, 1005, 1105, 1205, 1305, 1405, 1505, 1605,
        1705, 1805, 1905, 2005,
    ],
];

/// Token ID type
type TokenId = u32;

/// Generate random ASCII strings of given length
fn random_ascii_string(len: usize) -> String {
    Alphanumeric.sample_string(&mut thread_rng(), len)
}

/// Generate realistic LLM request texts
fn generate_realistic_requests(count: usize) -> Vec<String> {
    let mut rng = thread_rng();
    (0..count)
        .map(|_| {
            let prefix_idx = rng.random_range(0..CONVERSATION_PREFIXES.len());
            let query_len = rng.random_range(1000..3000);
            format!(
                "{}{}",
                CONVERSATION_PREFIXES[prefix_idx],
                random_ascii_string(query_len)
            )
        })
        .collect()
}

/// Generate random token sequences (simulating tokenized input)
fn generate_token_sequences(count: usize, len_range: (usize, usize)) -> Vec<Vec<TokenId>> {
    let mut rng = thread_rng();
    (0..count)
        .map(|_| {
            let len = rng.random_range(len_range.0..len_range.1);
            (0..len).map(|_| rng.random_range(0..50000)).collect()
        })
        .collect()
}

/// Generate worker endpoint URLs for scaling tests
fn generate_worker_endpoints(count: usize) -> Vec<String> {
    (0..count)
        .map(|i| {
            if i % 4 == 0 {
                format!("grpc://worker-{}.sglang.svc.cluster.local:50051", i)
            } else {
                format!("http://worker-{}.sglang.svc.cluster.local:8000", i)
            }
        })
        .collect()
}

/// Main benchmark comparing StringTree vs TokenTree
fn bench_summary(c: &mut Criterion) {
    let mut group = c.benchmark_group("benchmark_summary");

    // Reduce warmup and measurement time for faster runs
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(3));

    // Configuration constants
    const TREE_SIZE: usize = 10_000;
    const INSERT_POOL_SIZE: usize = 10_000;
    const NUM_THREADS: usize = 64;
    const OPS_PER_THREAD: usize = 200;
    const WORKER_COUNTS: [usize; 4] = [10, 50, 100, 500];

    // Pre-generate requests
    let string_requests = generate_realistic_requests(TREE_SIZE);
    let avg_chars: usize =
        string_requests.iter().map(|r| r.len()).sum::<usize>() / string_requests.len();

    // Token sequences with ~4 chars per token ratio
    let token_sequences = generate_token_sequences(TREE_SIZE, (250, 750));
    let avg_tokens: usize =
        token_sequences.iter().map(|s| s.len()).sum::<usize>() / token_sequences.len();

    // Report test configuration
    add_result("config", "Test Configuration:".to_string());
    add_result(
        "config",
        format!(
            "  StringTree: ~{} chars (~{} tokens equivalent)",
            avg_chars,
            avg_chars / 4
        ),
    );
    add_result("config", format!("  TokenTree:  ~{} tokens", avg_tokens));
    add_result(
        "config",
        format!("  Tree size: {} entries (for MATCH tests)", TREE_SIZE),
    );
    add_result(
        "config",
        format!("  Insert pool: {} unique requests", INSERT_POOL_SIZE),
    );
    add_result(
        "config",
        format!(
            "  Concurrency: {} threads x {} ops/thread",
            NUM_THREADS, OPS_PER_THREAD
        ),
    );
    add_result(
        "config",
        format!("  Worker counts tested: {:?}", WORKER_COUNTS),
    );

    // ========================================================================
    // OLD vs NEW StringTree Comparison
    // ========================================================================
    add_result("oldnew", "".to_string());
    add_result(
        "oldnew",
        "OLD (policies::tree) vs NEW (radix_tree::StringTree):".to_string(),
    );

    // Pre-populate both trees
    let old_tree = Arc::new(OldTree::new());
    let new_tree = Arc::new(StringTree::new());
    for (i, req) in string_requests.iter().enumerate() {
        let tenant = ENDPOINT_TENANTS[i % ENDPOINT_TENANTS.len()];
        old_tree.insert(req, tenant);
        new_tree.insert_text(req, tenant);
    }

    // Benchmark OLD tree match
    let printed_old = Arc::new(AtomicBool::new(false));
    let old_tree_clone = old_tree.clone();
    let requests_clone = string_requests.clone();
    group.bench_function("old_tree_match", |b| {
        let tree = old_tree_clone.clone();
        let requests = requests_clone.clone();
        let mut idx = 0;
        let printed = printed_old.clone();

        b.iter(|| {
            let result = tree.prefix_match(black_box(&requests[idx % requests.len()]));
            idx += 1;
            black_box(result)
        });

        if !printed.swap(true, Ordering::Relaxed) {
            // Results will be printed by Criterion
        }
    });

    // Benchmark NEW tree match
    let printed_new = Arc::new(AtomicBool::new(false));
    let new_tree_clone = new_tree.clone();
    let requests_clone = string_requests.clone();
    group.bench_function("new_tree_match", |b| {
        let tree = new_tree_clone.clone();
        let requests = requests_clone.clone();
        let mut idx = 0;
        let printed = printed_new.clone();

        b.iter(|| {
            let result = tree.prefix_match_legacy(black_box(&requests[idx % requests.len()]));
            idx += 1;
            black_box(result)
        });

        if !printed.swap(true, Ordering::Relaxed) {
            // Results will be printed by Criterion
        }
    });

    add_result(
        "oldnew",
        "  (See Criterion output above for OLD vs NEW timing)".to_string(),
    );

    // ========================================================================
    // StringTree vs TokenTree INSERT at different worker scales
    // ========================================================================
    let insert_string_requests = generate_realistic_requests(INSERT_POOL_SIZE);
    let insert_token_sequences = generate_token_sequences(INSERT_POOL_SIZE, (250, 750));

    for &num_workers in &WORKER_COUNTS {
        let workers = generate_worker_endpoints(num_workers);

        // StringTree INSERT
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("string_insert_{}w", num_workers);
        let insert_requests = insert_string_requests.clone();
        let workers_clone = workers.clone();

        group.bench_function(&bench_name, |b| {
            let workers = workers_clone.clone();
            let printed = printed.clone();
            let insert_requests = insert_requests.clone();

            b.iter_custom(|iters| {
                let tree = StringTree::new();
                let start = Instant::now();
                for i in 0..iters {
                    let tenant = &workers[i as usize % workers.len()];
                    let text = &insert_requests[i as usize % insert_requests.len()];
                    tree.insert_text(black_box(text), tenant);
                }
                let duration = start.elapsed();

                if !printed.load(Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let throughput_mb = (ops_per_sec * avg_chars as f64) / 1_000_000.0;
                    add_result(
                        "string",
                        format!(
                            "INSERT {:>3} workers: {:>8.0} ops/sec | {:>5.1} µs/op | {:>6.1} MB/s | ~{} chars",
                            num_workers, ops_per_sec, latency_us, throughput_mb, avg_chars
                        ),
                    );
                    printed.store(true, Ordering::Relaxed);
                }

                duration
            });
        });

        // TokenTree INSERT
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("token_insert_{}w", num_workers);
        let insert_seqs = insert_token_sequences.clone();
        let workers_clone = workers.clone();

        group.bench_function(&bench_name, |b| {
            let workers = workers_clone.clone();
            let printed = printed.clone();
            let insert_seqs = insert_seqs.clone();

            b.iter_custom(|iters| {
                let tree = TokenTree::new();
                let start = Instant::now();
                for i in 0..iters {
                    let tenant = &workers[i as usize % workers.len()];
                    let tokens = &insert_seqs[i as usize % insert_seqs.len()];
                    tree.insert_tokens(black_box(tokens), tenant);
                }
                let duration = start.elapsed();

                if !printed.load(Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let throughput_mtok = (ops_per_sec * avg_tokens as f64) / 1_000_000.0;
                    add_result(
                        "token",
                        format!(
                            "INSERT {:>3} workers: {:>8.0} ops/sec | {:>5.1} µs/op | {:>6.1} Mtok/s | ~{} tokens",
                            num_workers, ops_per_sec, latency_us, throughput_mtok, avg_tokens
                        ),
                    );
                    printed.store(true, Ordering::Relaxed);
                }

                duration
            });
        });
    }

    // ========================================================================
    // StringTree vs TokenTree MATCH at different worker scales
    // ========================================================================
    for &num_workers in &WORKER_COUNTS {
        let workers = generate_worker_endpoints(num_workers);

        // Pre-populate StringTree
        let string_tree = Arc::new(StringTree::new());
        for (i, req) in string_requests.iter().enumerate() {
            let tenant = &workers[i % workers.len()];
            string_tree.insert_text(req, tenant);
        }

        // Pre-populate TokenTree
        let token_tree = Arc::new(TokenTree::new());
        for (i, seq) in token_sequences.iter().enumerate() {
            let tenant = &workers[i % workers.len()];
            token_tree.insert_tokens(seq, tenant);
        }

        // StringTree MATCH
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("string_match_{}w", num_workers);
        let tree_clone = string_tree.clone();
        let requests_clone = string_requests.clone();

        group.bench_function(&bench_name, |b| {
            let tree = tree_clone.clone();
            let requests = requests_clone.clone();
            let mut idx = 0;
            let printed = printed.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let result = tree.prefix_match_legacy(black_box(&requests[idx % requests.len()]));
                    black_box(result);
                    idx += 1;
                }
                let duration = start.elapsed();

                if !printed.load(Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let throughput_mb = (ops_per_sec * avg_chars as f64) / 1_000_000.0;
                    add_result(
                        "string",
                        format!(
                            "MATCH  {:>3} workers: {:>8.0} ops/sec | {:>5.1} µs/op | {:>6.1} MB/s | {}k tree entries",
                            num_workers, ops_per_sec, latency_us, throughput_mb, TREE_SIZE / 1000
                        ),
                    );
                    printed.store(true, Ordering::Relaxed);
                }

                duration
            });
        });

        // TokenTree MATCH
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("token_match_{}w", num_workers);
        let tree_clone = token_tree.clone();
        let seqs_clone = token_sequences.clone();

        group.bench_function(&bench_name, |b| {
            let tree = tree_clone.clone();
            let seqs = seqs_clone.clone();
            let mut idx = 0;
            let printed = printed.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let result = tree.prefix_match_legacy(black_box(&seqs[idx % seqs.len()]));
                    black_box(result);
                    idx += 1;
                }
                let duration = start.elapsed();

                if !printed.load(Ordering::Relaxed) {
                    let ops_per_sec = iters as f64 / duration.as_secs_f64();
                    let latency_us = duration.as_nanos() as f64 / iters as f64 / 1000.0;
                    let throughput_mtok = (ops_per_sec * avg_tokens as f64) / 1_000_000.0;
                    add_result(
                        "token",
                        format!(
                            "MATCH  {:>3} workers: {:>8.0} ops/sec | {:>5.1} µs/op | {:>6.1} Mtok/s | {}k tree entries",
                            num_workers, ops_per_sec, latency_us, throughput_mtok, TREE_SIZE / 1000
                        ),
                    );
                    printed.store(true, Ordering::Relaxed);
                }

                duration
            });
        });
    }

    // ========================================================================
    // CONCURRENT benchmarks
    // ========================================================================
    group.sample_size(10);

    for &num_workers in &WORKER_COUNTS {
        let workers = generate_worker_endpoints(num_workers);

        // StringTree CONCURRENT
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("string_concurrent_{}w", num_workers);
        let workers_clone = workers.clone();

        group.bench_function(&bench_name, |b| {
            let printed = printed.clone();
            let workers = workers_clone.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let tree = Arc::new(StringTree::new());
                    let workers_ref = &workers;
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let worker = workers_ref[t % workers_ref.len()].clone();
                            thread::spawn(move || {
                                for i in 0..OPS_PER_THREAD {
                                    let text = format!(
                                        "{}thread{}_request{}",
                                        CONVERSATION_PREFIXES[i % CONVERSATION_PREFIXES.len()],
                                        t,
                                        i
                                    );
                                    if i % 3 == 0 {
                                        tree.prefix_match_legacy(&text);
                                    } else {
                                        tree.insert_text(&text, &worker);
                                    }
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                }
                let duration = start.elapsed();

                if !printed.load(Ordering::Relaxed) {
                    let total_ops = iters * NUM_THREADS as u64 * OPS_PER_THREAD as u64;
                    let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                    let per_thread_ops = ops_per_sec / NUM_THREADS as f64;
                    add_result(
                        "string",
                        format!(
                            "CONCURRENT {:>3} workers: {:>6.0} ops/sec | {} threads | {:.0} ops/thread",
                            num_workers, ops_per_sec, NUM_THREADS, per_thread_ops
                        ),
                    );
                    printed.store(true, Ordering::Relaxed);
                }

                duration
            });
        });

        // TokenTree CONCURRENT
        let printed = Arc::new(AtomicBool::new(false));
        let bench_name = format!("token_concurrent_{}w", num_workers);
        let workers_clone = workers.clone();

        group.bench_function(&bench_name, |b| {
            let printed = printed.clone();
            let workers = workers_clone.clone();

            b.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    let tree = Arc::new(TokenTree::new());
                    let workers_ref = &workers;
                    let handles: Vec<_> = (0..NUM_THREADS)
                        .map(|t| {
                            let tree = Arc::clone(&tree);
                            let worker = workers_ref[t % workers_ref.len()].clone();
                            thread::spawn(move || {
                                for i in 0..OPS_PER_THREAD {
                                    // Use shared prefix + unique suffix (mirrors StringTree's CONVERSATION_PREFIXES pattern)
                                    let prefix = &TOKEN_PREFIXES[i % TOKEN_PREFIXES.len()];
                                    let mut tokens: Vec<TokenId> = prefix.to_vec();
                                    // Add unique suffix tokens
                                    tokens.extend((0..30).map(|j| (t * 10000 + i * 100 + j) as u32));
                                    if i % 3 == 0 {
                                        tree.prefix_match_legacy(&tokens);
                                    } else {
                                        tree.insert_tokens(&tokens, &worker);
                                    }
                                }
                            })
                        })
                        .collect();

                    for h in handles {
                        h.join().unwrap();
                    }
                }
                let duration = start.elapsed();

                if !printed.load(Ordering::Relaxed) {
                    let total_ops = iters * NUM_THREADS as u64 * OPS_PER_THREAD as u64;
                    let ops_per_sec = total_ops as f64 / duration.as_secs_f64();
                    let per_thread_ops = ops_per_sec / NUM_THREADS as f64;
                    add_result(
                        "token",
                        format!(
                            "CONCURRENT {:>3} workers: {:>6.0} ops/sec | {} threads | {:.0} ops/thread",
                            num_workers, ops_per_sec, NUM_THREADS, per_thread_ops
                        ),
                    );
                    printed.store(true, Ordering::Relaxed);
                }

                duration
            });
        });
    }

    group.finish();
}

/// Print final summary table
fn print_summary() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();

    eprintln!("\n{}", "=".repeat(100));
    eprintln!("RADIX TREE BENCHMARK SUMMARY (StringTree vs TokenTree)");
    eprintln!("{}", "=".repeat(100));

    let results = BENCHMARK_RESULTS.lock().unwrap();
    eprintln!("Total benchmark results collected: {}", results.len());

    // Collect results by category
    let mut config_results = Vec::new();
    let mut oldnew_results = Vec::new();
    let mut string_results = Vec::new();
    let mut token_results = Vec::new();

    for (key, value) in results.iter() {
        let category = key.split('_').skip(1).collect::<Vec<_>>().join("_");
        match category.as_str() {
            "config" => config_results.push(value.clone()),
            "oldnew" => oldnew_results.push(value.clone()),
            "string" => string_results.push(value.clone()),
            "token" => token_results.push(value.clone()),
            _ => {}
        }
    }

    // Print config
    eprintln!("\n{}", "-".repeat(100));
    eprintln!("TEST CONFIGURATION");
    eprintln!("{}", "-".repeat(100));
    for v in &config_results {
        eprintln!("{}", v);
    }

    // Print old vs new
    eprintln!("\n{}", "-".repeat(100));
    eprintln!("OLD vs NEW STRINGTREE VALIDATION");
    eprintln!("{}", "-".repeat(100));
    for v in &oldnew_results {
        eprintln!("{}", v);
    }

    // Print StringTree results
    eprintln!("\n{}", "-".repeat(100));
    eprintln!("STRINGTREE BENCHMARK RESULTS (Character-based, HTTP router)");
    eprintln!("{}", "-".repeat(100));
    for v in &string_results {
        eprintln!("{}", v);
    }

    // Print TokenTree results
    eprintln!("\n{}", "-".repeat(100));
    eprintln!("TOKENTREE BENCHMARK RESULTS (Token-based, gRPC router)");
    eprintln!("{}", "-".repeat(100));
    for v in &token_results {
        eprintln!("{}", v);
    }

    eprintln!("\n{}", "=".repeat(100));
    eprintln!("Endpoint tenants used:");
    for (i, tenant) in ENDPOINT_TENANTS.iter().enumerate() {
        eprintln!("  [{}] {}", i, tenant);
    }
    eprintln!("{}", "=".repeat(100));
}

fn run_benchmarks(c: &mut Criterion) {
    bench_summary(c);
    print_summary();
}

criterion_group!(benches, run_benchmarks);
criterion_main!(benches);
