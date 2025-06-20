#!/usr/bin/env python3

import pandas as pd
import httpx
import asyncio
import time
import os
import gc
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from urllib.parse import urlparse
from typing import Dict, Optional, List
import sys
from tqdm import tqdm
import concurrent.futures
import psutil
import threading
import warnings
from collections import defaultdict
import random
import socket

# Disable XML parsing warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# DNS Cache to reduce DNS lookups
DNS_CACHE = {}

async def resolve_domain_cached(domain: str) -> Optional[str]:
    """Resolve domain with caching to reduce DNS bottleneck"""
    try:
        parsed = urlparse(domain if domain.startswith(('http://', 'https://')) else f'https://{domain}')
        hostname = parsed.netloc or parsed.path.split('/')[0]
        
        if hostname in DNS_CACHE:
            return DNS_CACHE[hostname]
            
        # Resolve DNS
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, socket.gethostbyname, hostname)
        DNS_CACHE[hostname] = result
        return result
    except Exception:
        DNS_CACHE[hostname] = None
        return None

class PerformanceDiagnostics:
    def __init__(self):
        self.stats = {
            'dns_time': [],
            'connect_time': [],
            'response_time': [],
            'parse_time': [],
            'status_codes': defaultdict(int),
            'error_types': defaultdict(int),
            'successful_requests': 0,
            'failed_requests': 0,
            'rate_limited_count': 0,
            'dns_cache_hits': 0,
            'dns_cache_misses': 0
        }
        
    def record_request(self, domain, start_time, end_time, status_code=None, error_type=None):
        """Record performance metrics for a request"""
        total_time = end_time - start_time
        
        if status_code:
            self.stats['status_codes'][status_code] += 1
            if status_code == 429:
                self.stats['rate_limited_count'] += 1
            self.stats['successful_requests'] += 1
            self.stats['response_time'].append(total_time)
        else:
            self.stats['error_types'][error_type] += 1
            self.stats['failed_requests'] += 1
            
    def record_dns_cache(self, hit: bool):
        """Record DNS cache performance"""
        if hit:
            self.stats['dns_cache_hits'] += 1
        else:
            self.stats['dns_cache_misses'] += 1
            
    def get_bottleneck_analysis(self):
        """Analyze performance data to identify bottlenecks"""
        if not self.stats['response_time']:
            return "No performance data available"
            
        avg_response_time = sum(self.stats['response_time']) / len(self.stats['response_time'])
        success_rate = self.stats['successful_requests'] / (self.stats['successful_requests'] + self.stats['failed_requests']) * 100
        total_dns_requests = self.stats['dns_cache_hits'] + self.stats['dns_cache_misses']
        dns_cache_rate = (self.stats['dns_cache_hits'] / total_dns_requests * 100) if total_dns_requests > 0 else 0
        
        analysis = []
        analysis.append(f"📊 Performance Analysis:")
        analysis.append(f"   Average response time: {avg_response_time:.2f}s")
        analysis.append(f"   Success rate: {success_rate:.1f}%")
        analysis.append(f"   Total requests: {self.stats['successful_requests'] + self.stats['failed_requests']}")
        analysis.append(f"   Rate limited (429): {self.stats['rate_limited_count']}")
        analysis.append(f"   DNS cache hit rate: {dns_cache_rate:.1f}%")
        
        # Status code analysis
        if self.stats['status_codes']:
            analysis.append(f"   Status codes: {dict(self.stats['status_codes'])}")
            
        # Error analysis
        if self.stats['error_types']:
            analysis.append(f"   Error types: {dict(self.stats['error_types'])}")
            
        # Bottleneck identification with solutions
        if avg_response_time > 10:
            analysis.append("🚧 BOTTLENECK: Slow server responses (avg >10s)")
            analysis.append("   💡 SOLUTION: Reduce timeout, skip slow domains")
        if success_rate < 70:
            analysis.append("🚧 BOTTLENECK: High failure rate - possible rate limiting")
            analysis.append("   💡 SOLUTION: Add delays, rotate user agents, reduce workers")
        if self.stats['rate_limited_count'] > self.stats['successful_requests'] * 0.1:
            analysis.append("🚧 BOTTLENECK: Rate limiting detected (many 429s)")
            analysis.append("   💡 SOLUTION: Increase delays, reduce concurrent requests")
        if self.stats['error_types'].get('Request', 0) > self.stats['successful_requests'] * 0.1:
            analysis.append("🚧 BOTTLENECK: DNS/Network issues")
            analysis.append(f"   💡 SOLUTION: DNS caching active ({dns_cache_rate:.1f}% hit rate)")
        if self.stats['error_types'].get('Timeout', 0) > self.stats['successful_requests'] * 0.2:
            analysis.append("🚧 BOTTLENECK: Network timeouts")
            analysis.append("   💡 SOLUTION: Shorter timeouts, retry fast domains")
            
        return '\n'.join(analysis)

class ResourceMonitor:
    def __init__(self):
        self.monitoring = False
        self.live_updates = False
        self.stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'memory_mb': [],
            'network_connections': [],
            'network_bytes_sent': [],
            'network_bytes_recv': [],
            'timestamps': []
        }
        self.monitor_thread = None
        self.live_update_thread = None
        self.last_network_io = None
        
    def start_monitoring(self, live_updates=True):
        """Start resource monitoring in background thread"""
        self.monitoring = True
        self.live_updates = live_updates
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        if live_updates:
            self.live_update_thread = threading.Thread(target=self._live_update_loop, daemon=True)
            self.live_update_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        self.live_updates = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        if self.live_update_thread:
            self.live_update_thread.join(timeout=1)
            
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Get system stats
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                network_connections = len(psutil.net_connections())
                network_io = psutil.net_io_counters()
                
                # Calculate network throughput
                if self.last_network_io:
                    bytes_sent_rate = network_io.bytes_sent - self.last_network_io.bytes_sent
                    bytes_recv_rate = network_io.bytes_recv - self.last_network_io.bytes_recv
                else:
                    bytes_sent_rate = 0
                    bytes_recv_rate = 0
                    
                self.last_network_io = network_io
                
                # Store stats
                self.stats['cpu_percent'].append(cpu_percent)
                self.stats['memory_percent'].append(memory.percent)
                self.stats['memory_mb'].append(memory.used / 1024 / 1024)
                self.stats['network_connections'].append(network_connections)
                self.stats['network_bytes_sent'].append(bytes_sent_rate)
                self.stats['network_bytes_recv'].append(bytes_recv_rate)
                self.stats['timestamps'].append(time.time())
                
                # Keep only last 60 readings (1 minute of data)
                for key in self.stats:
                    if len(self.stats[key]) > 60:
                        self.stats[key] = self.stats[key][-60:]
                        
            except Exception:
                pass
                
    def _live_update_loop(self):
        """Live update loop - prints stats every 1 second"""
        while self.live_updates:
            try:
                time.sleep(1)  # Update every 1 second (changed from 5)
                if self.live_updates and self.stats['cpu_percent']:
                    stats = self.get_current_stats()
                    if stats:
                        # Clear the line and print resource stats
                        print(f"\r🔄 Live Stats: CPU {stats['cpu_avg']:.1f}% | Memory {stats['memory_current']:.1f}% | Connections {stats['connections']} | Network ↑{stats['net_up']:.1f}MB/s ↓{stats['net_down']:.1f}MB/s", end='', flush=True)
            except Exception:
                pass
                
    def get_current_stats(self):
        """Get current resource usage"""
        if not self.stats['cpu_percent']:
            return None
            
        return {
            'cpu_avg': sum(self.stats['cpu_percent'][-10:]) / min(10, len(self.stats['cpu_percent'])),
            'cpu_max': max(self.stats['cpu_percent'][-10:]) if self.stats['cpu_percent'] else 0,
            'memory_current': self.stats['memory_percent'][-1] if self.stats['memory_percent'] else 0,
            'memory_mb': self.stats['memory_mb'][-1] if self.stats['memory_mb'] else 0,
            'connections': self.stats['network_connections'][-1] if self.stats['network_connections'] else 0,
            'net_up': (self.stats['network_bytes_sent'][-1] / 1024 / 1024) if self.stats['network_bytes_sent'] else 0,
            'net_down': (self.stats['network_bytes_recv'][-1] / 1024 / 1024) if self.stats['network_bytes_recv'] else 0
        }
        
    def get_recommendations(self, current_workers):
        """Get worker count recommendations based on resource usage"""
        stats = self.get_current_stats()
        if not stats:
            return "Insufficient data for recommendations"
            
        recommendations = []
        
        # CPU recommendations
        if stats['cpu_avg'] < 40:
            recommendations.append("✓ CPU usage low - not the bottleneck")
        elif stats['cpu_avg'] > 85:
            recommendations.append("⚠ CPU usage high - CPU bottleneck detected")
        else:
            recommendations.append("✓ CPU usage optimal")
            
        # Memory recommendations  
        if stats['memory_current'] < 70:
            recommendations.append("✓ Memory usage low - not the bottleneck")
        elif stats['memory_current'] > 90:
            recommendations.append("⚠ Memory usage high - memory bottleneck")
        else:
            recommendations.append("✓ Memory usage optimal")
            
        # Network recommendations
        if stats['connections'] < 300:
            recommendations.append("✓ Network connections low")
        elif stats['connections'] > 800:
            recommendations.append("⚠ Too many network connections")
        else:
            recommendations.append("✓ Network connections optimal")
            
        # Network throughput analysis
        total_throughput = stats['net_up'] + stats['net_down']
        if total_throughput > 50:  # >50 MB/s
            recommendations.append("⚠ High network throughput - bandwidth bottleneck possible")
        elif total_throughput < 5:  # <5 MB/s
            recommendations.append("✓ Low network usage - not bandwidth limited")
            
        # Bottleneck detection
        if stats['cpu_avg'] < 50 and stats['memory_current'] < 70 and total_throughput < 20:
            recommendations.append("🚧 LIKELY BOTTLENECK: External factors (DNS, rate limiting, server response times)")
        elif stats['cpu_avg'] > 80:
            recommendations.append("🚧 LIKELY BOTTLENECK: CPU (BeautifulSoup parsing)")
        elif total_throughput > 30:
            recommendations.append("🚧 LIKELY BOTTLENECK: Network bandwidth")
        else:
            recommendations.append(f"💡 System resources available - bottleneck is external")
            
        return recommendations

class DomainScraper:
    def __init__(self, input_file: str, max_workers: int = 10):
        self.input_file = input_file
        self.output_file = input_file.replace('.parquet', '_metadata.parquet')
        self.max_workers = max_workers
        self.processed_count = 0
        self.start_time = time.time()
        self.resource_monitor = ResourceMonitor()
        self.performance_diagnostics = PerformanceDiagnostics()
        # Shared HTTP client for all workers
        self.shared_client = None
        # Rate limiting mitigation
        self.request_delay = 0.1  # Start with 100ms delay
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
        
    async def initialize_client(self):
        """Initialize shared HTTP client with optimized connection pool"""
        self.shared_client = httpx.AsyncClient(
            timeout=10.0,  # Reduced from 20s for faster failures
            follow_redirects=True,
            verify=False,
            limits=httpx.Limits(
                max_connections=500,  # Total connection pool for all workers
                max_keepalive_connections=100,  # Keep some connections alive for reuse
                keepalive_expiry=20.0  # Keep connections alive for 1 minute
            ),
            headers={
                'User-Agent': random.choice(self.user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        )
        
    async def cleanup_client(self):
        """Clean up shared HTTP client"""
        if self.shared_client:
            await self.shared_client.aclose()
        
    async def scrape_domain(self, domain: str, domain_times_map: dict = None) -> Dict[str, Optional[str]]:
        """Scrape title and meta description from a domain using shared client with optimizations"""
        start_time = time.time()
        error_type = None
        status_code = None

        # Store the original domain and times exactly as they appear in input
        original_domain = domain  # Keep exactly as is from parquet
        domain_times = domain_times_map.get(original_domain) if domain_times_map else None

        try:
            # Clean the domain for the actual HTTP request (but keep original for output)
            clean_domain = domain.strip()
            if clean_domain.startswith('@'):
                clean_domain = clean_domain[1:].strip()
            
            # Ensure clean domain has protocol
            if not clean_domain.startswith(('http://', 'https://')):
                clean_domain = f'https://{clean_domain}'
            
            # DNS caching optimization
            hostname = urlparse(clean_domain).netloc
            if hostname in DNS_CACHE:
                self.performance_diagnostics.record_dns_cache(True)
            else:
                self.performance_diagnostics.record_dns_cache(False)
                # Pre-resolve DNS to cache it
                await resolve_domain_cached(clean_domain)
            
            # Rate limiting mitigation - random delay
            if self.request_delay > 0:
                delay = self.request_delay + random.uniform(0, self.request_delay)
                await asyncio.sleep(delay)
            
            # Rotate user agent for each request
            headers = {'User-Agent': random.choice(self.user_agents)}
            
            # Use shared client with shorter timeout
            try:
                # First try with HTTPS
                try:
                    response = await self.shared_client.get(
                        clean_domain, 
                        timeout=10.0,
                        headers=headers,
                        follow_redirects=True  # Ensure redirects are followed
                    )
                except Exception as e:
                    # If HTTPS fails, try HTTP
                    if clean_domain.startswith('https://'):
                        http_domain = clean_domain.replace('https://', 'http://')
                        response = await self.shared_client.get(
                            http_domain,
                            timeout=10.0,
                            headers=headers,
                            follow_redirects=True
                        )
                    else:
                        raise  # Re-raise if it wasn't an HTTPS issue

                status_code = response.status_code
                final_url = str(response.url)

                # If we got redirected, record the redirect chain
                redirect_url = None
                if response.history:
                    # Record the redirect chain status codes
                    for r in response.history:
                        self.performance_diagnostics.record_request(clean_domain, start_time, time.time(), r.status_code)
                    
                    # Store the final URL as redirect_url if different from clean domain
                    if final_url != clean_domain:
                        redirect_url = final_url

                # Get content with encoding handling
                try:
                    content = response.content[:1000000]  # First try binary content
                    if not content:
                        content = response.text.encode('utf-8')[:1000000]  # Try text content
                except Exception:
                    content = b''  # Empty content as last resort

                self._last_status_code = status_code  # Track for adaptive scaling
                
                # Handle rate limiting first (before raise_for_status)
                if status_code == 429:
                    # Increase delay for future requests
                    self.request_delay = min(self.request_delay * 1.5, 2.0)  # Max 2 seconds
                    self.performance_diagnostics.record_request(clean_domain, start_time, time.time(), status_code)
                    result = {
                        'domain': original_domain,  # Use original domain from parquet
                        'redirect_url': redirect_url,
                        'status_code': status_code,
                        'title': None,
                        'meta_description': None,
                        'keywords': None,
                        'og_title': None,
                        'og_description': None,
                        'og_type': None
                    }
                    if domain_times is not None:
                        result['domain_times'] = domain_times
                    return result

                # Handle successful responses (200-299 range)
                if 200 <= status_code < 300:
                    # Reduce delay on success
                    self.request_delay = max(self.request_delay * 0.95, 0.05)  # Min 50ms
                    
                    # Parse HTML with timeout protection - using proven scraping technique
                    try:
                        # Try to parse with BeautifulSoup, handling encoding issues gracefully
                        try:
                            soup = BeautifulSoup(content, 'html.parser')
                        except Exception:
                            # Fallback: try with response.text if content parsing fails
                            soup = BeautifulSoup(response.text[:1000000], 'html.parser')
                        
                        # Extract title - using proven method from tester (.get_text() works better than .string)
                        title = None
                        title_tag = soup.find('title')
                        if title_tag:
                            title = title_tag.get_text().strip()[:500]  # Limit title length
                        
                        # Extract meta description and other tags
                        meta_description = None
                        keywords = None
                        og_title = None
                        og_description = None
                        og_type = None
                        
                        try:
                            # Get meta description
                            meta_tag = soup.find('meta', attrs={'name': 'description'})
                            if meta_tag and hasattr(meta_tag, 'get'):
                                content = meta_tag.get('content', '')
                                if content:
                                    meta_description = str(content).strip()[:1000]  # Limit description length
                            
                            # Get meta keywords
                            keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
                            if keywords_tag and hasattr(keywords_tag, 'get'):
                                content = keywords_tag.get('content', '')
                                if content:
                                    keywords = str(content).strip()[:1000]  # Limit keywords length
                            
                            # Get specific Open Graph tags
                            og_title_tag = soup.find('meta', attrs={'property': 'og:title'})
                            if og_title_tag and hasattr(og_title_tag, 'get'):
                                content = og_title_tag.get('content', '')
                                if content:
                                    og_title = str(content).strip()[:500]

                            og_desc_tag = soup.find('meta', attrs={'property': 'og:description'})
                            if og_desc_tag and hasattr(og_desc_tag, 'get'):
                                content = og_desc_tag.get('content', '')
                                if content:
                                    og_description = str(content).strip()[:1000]

                            og_type_tag = soup.find('meta', attrs={'property': 'og:type'})
                            if og_type_tag and hasattr(og_type_tag, 'get'):
                                content = og_type_tag.get('content', '')
                                if content:
                                    og_type = str(content).strip()[:100]
                            
                            # Fallback to og:description if no standard meta description found
                            if not meta_description and og_description:
                                meta_description = og_description
                                
                        except Exception as e:
                            # Log meta tag parsing error but continue
                            pass
                        
                        # Record successful request
                        end_time = time.time()
                        self.performance_diagnostics.record_request(clean_domain, start_time, end_time, status_code)
                        
                        result = {
                            'domain': original_domain,  # Use original domain from parquet
                            'redirect_url': redirect_url,
                            'status_code': status_code,
                            'title': title,
                            'meta_description': meta_description,
                            'keywords': keywords,
                            'og_title': og_title,
                            'og_description': og_description,
                            'og_type': og_type
                        }
                        if domain_times is not None:
                            result['domain_times'] = domain_times
                        
                        # Debug logging for empty results (uncomment if needed for debugging)
                        if not title and not meta_description:
                            print(f"DEBUG: Empty result for {original_domain} - HTML size: {len(content)} bytes - Status: {status_code}")
                        
                        return result
                        
                    except Exception as e:
                        # HTML parsing failed but we got the response - log the error
                        print(f"DEBUG: HTML parsing failed for {original_domain}: {type(e).__name__}: {str(e)}")
                        end_time = time.time()
                        self.performance_diagnostics.record_request(clean_domain, start_time, end_time, status_code)
                else:
                    # Record non-200 status
                    end_time = time.time()
                    self.performance_diagnostics.record_request(clean_domain, start_time, end_time, status_code)
                
            except asyncio.TimeoutError:
                error_type = "Timeout"
                status_code = None
            except httpx.HTTPStatusError as e:
                error_type = f"HTTP_{e.response.status_code}"
                status_code = e.response.status_code if hasattr(e, 'response') else None
            except httpx.ConnectError:
                error_type = "Connection"
                status_code = None
            except httpx.UnsupportedProtocol:
                error_type = "Protocol"
                status_code = None
            except httpx.InvalidURL:
                error_type = "InvalidURL"
                status_code = None
            except httpx.RequestError:
                error_type = "Request"
                status_code = None
            except Exception as e:
                error_type = f"Other: {type(e).__name__}"
                status_code = None
                
        except asyncio.TimeoutError:
            error_type = "Timeout"
            status_code = None
        except httpx.HTTPStatusError as e:
            error_type = f"HTTP_{e.response.status_code}"
            status_code = e.response.status_code if hasattr(e, 'response') else None
        except httpx.ConnectError:
            error_type = "Connection"
            status_code = None
        except httpx.UnsupportedProtocol:
            error_type = "Protocol"
            status_code = None
        except httpx.InvalidURL:
            error_type = "InvalidURL"
            status_code = None
        except httpx.RequestError:
            error_type = "Request"
            status_code = None
        except Exception as e:
            error_type = f"Other: {type(e).__name__}"
            status_code = None
            
        # Record failed request
        end_time = time.time()
        self.performance_diagnostics.record_request(clean_domain, start_time, end_time, error_type=error_type)
        
        result = {
            'domain': original_domain,  # Use original domain from parquet
            'redirect_url': None,  # No redirect for errors
            'status_code': status_code,  # Include status code even for errors
            'title': None,
            'meta_description': None,
            'keywords': None,
            'og_title': None,
            'og_description': None,
            'og_type': None
        }
        if domain_times is not None:
            result['domain_times'] = domain_times
        return result
    
    async def process_domains_concurrent(self, domains: List[str], chunk_start: int, main_pbar=None, domain_times_map=None):
        """Process domains concurrently while maintaining original order"""
        semaphore = asyncio.Semaphore(self.max_workers)
        
        # Track performance for monitoring
        chunk_start_time = time.time()
        successful_in_chunk = 0
        
        # Create progress bar for this chunk
        chunk_pbar = tqdm(
            total=len(domains),
            desc=f"Chunk Progress",
            position=2,
            leave=False,
            unit="domain",
            mininterval=1,
            maxinterval=1
        )
        
        async def process_single_domain_with_index(index: int, domain: str):
            nonlocal successful_in_chunk
            async with semaphore:
                result = await self.scrape_domain(domain, domain_times_map)
                
                # Track chunk performance
                if result['title'] is not None or result['meta_description'] is not None:
                    successful_in_chunk += 1
                
                # Update both progress bars immediately
                chunk_pbar.update(1)
                if main_pbar:
                    main_pbar.update(1)
                
                # Update chunk progress bar postfix
                chunk_pbar.set_postfix({
                    'Success': f'{successful_in_chunk}/{chunk_pbar.n}',
                    'Workers': self.max_workers,
                    'Delay': f'{self.request_delay:.2f}s'
                })
                
                # Update overall progress bar postfix
                if main_pbar:
                    elapsed = time.time() - self.start_time
                    speed = main_pbar.n / elapsed if elapsed > 0 else 0
                    main_pbar.set_postfix({
                        'Speed': f'{speed:.1f} dom/s',
                        'Workers': self.max_workers,
                        'Success': f'{successful_in_chunk}/{chunk_pbar.n}'
                    })
                
                return index, result
        
        # Create tasks for all domains with their indices
        tasks = [process_single_domain_with_index(i, domain) for i, domain in enumerate(domains)]
        
        # Store results with their original indices
        indexed_results = []
        
        # Process with progress tracking
        for completed_task in asyncio.as_completed(tasks):
            index, result = await completed_task
            indexed_results.append((index, result))
            self.processed_count += 1
        
        chunk_pbar.close()
        
        # Sort results by original index to maintain order
        indexed_results.sort(key=lambda x: x[0])
        
        # Extract just the results in original order
        results = [result for index, result in indexed_results]
        
        return results
    
    def save_progress(self, results: list):
        """Save results to parquet file with improved error handling"""
        if not results:
            print("⚠️ No results to save")
            return
            
        try:
            df = pd.DataFrame(results)
            
            # Ensure we have the expected columns in the correct order - INCLUDING redirect_url
            expected_columns = ['domain', 'domain_times', 'redirect_url', 'status_code', 'title', 'meta_description', 'keywords', 
                              'og_title', 'og_description', 'og_type']
            
            # Only include domain_times if it exists in the data
            if 'domain_times' not in df.columns:
                expected_columns = ['domain', 'redirect_url', 'status_code', 'title', 'meta_description', 'keywords',
                                  'og_title', 'og_description', 'og_type']
            
            # Only include redirect_url if it exists in the data
            if 'redirect_url' not in df.columns:
                expected_columns = [col for col in expected_columns if col != 'redirect_url']
            
            # Reorder columns to ensure consistency
            df = df[expected_columns]
            
            # Save to parquet
            df.to_parquet(self.output_file, index=False)
            print(f"💾 Saved {len(results):,} results to {self.output_file}")
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            # Try to save as CSV as backup
            try:
                backup_file = self.output_file.replace('.parquet', '_backup.csv')
                df = pd.DataFrame(results)
                df.to_csv(backup_file, index=False)
                print(f"💾 Backup saved as CSV: {backup_file}")
            except Exception as e2:
                print(f"❌ Failed to save backup: {e2}")
    
    async def reset_chunk_resources(self):
        """Reset resources after each chunk to prevent exponential slowdown"""
        global DNS_CACHE
        
        # Get memory before cleanup
        memory_before = psutil.virtual_memory().used / 1024 / 1024  # MB
        
        # 1. Clear DNS cache (keeps growing infinitely)
        dns_cache_size = len(DNS_CACHE)
        DNS_CACHE.clear()
        
        # 2. Reset rate limiting delay (keeps increasing)
        original_delay = self.request_delay
        self.request_delay = 0.1  # Reset to initial 100ms
        
        # 3. Reset performance diagnostics (response_time list grows forever)
        old_requests = self.performance_diagnostics.stats['successful_requests'] + self.performance_diagnostics.stats['failed_requests']
        self.performance_diagnostics = PerformanceDiagnostics()
        
        # 4. Clear resource monitor accumulated stats (arrays keep growing)
        if hasattr(self.resource_monitor, 'stats'):
            for key in self.resource_monitor.stats:
                if isinstance(self.resource_monitor.stats[key], list):
                    self.resource_monitor.stats[key] = self.resource_monitor.stats[key][-5:]  # Keep only last 5 readings
        
        # 5. COMPLETE HTTP CLIENT RESET - This is the main issue
        if self.shared_client:
            await self.shared_client.aclose()
            self.shared_client = None
        
        # Recreate fresh client with clean connection pool
        await self.initialize_client()
        
        # 6. Aggressive garbage collection and memory cleanup
        collected = gc.collect()  # Force garbage collection
        
        # 7. Clear any lingering variables and reset counters
        if hasattr(self, '_last_status_code'):
            delattr(self, '_last_status_code')
        
        # 8. System-level cleanup
        import os
        try:
            # Force close any lingering file descriptors
            os.sync()  # Flush filesystem buffers
        except:
            pass
        
        # 9. Additional memory pressure relief
        try:
            # Force Python to release memory back to OS
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass  # Ignore if not available
        
        # Get memory after cleanup
        memory_after = psutil.virtual_memory().used / 1024 / 1024  # MB
        memory_freed = memory_before - memory_after
        
        print(f"🔄 FULL RESET: DNS cache ({dns_cache_size} entries), delay ({original_delay:.3f}s → 0.1s), HTTP client recreated, stats ({old_requests} requests), GC collected {collected} objects, memory freed: {memory_freed:.1f}MB")
    
    def load_existing_data(self) -> Dict[str, Dict]:
        """Load existing processed data"""
        existing_data = {}
        if os.path.exists(self.output_file):
            try:
                df = pd.read_parquet(self.output_file)
                for _, row in df.iterrows():
                    existing_data[row['domain']] = {
                        'title': row.get('title'),
                        'meta_description': row.get('meta_description')
                    }
                print(f"Loaded {len(existing_data)} existing records")
            except Exception as e:
                print(f"Error loading existing data: {e}")
        return existing_data
    
    def print_resource_stats(self):
        """Print current resource usage and recommendations"""
        stats = self.resource_monitor.get_current_stats()
        if stats:
            # Clear the live update line first
            print(f"\r{' ' * 120}\r", end='')  # Clear the line
            
            print(f"\n📊 Resource Usage:")
            print(f"   CPU: {stats['cpu_avg']:.1f}% avg, {stats['cpu_max']:.1f}% max")
            print(f"   Memory: {stats['memory_current']:.1f}% ({stats['memory_mb']:.0f} MB)")
            print(f"   Network: {stats['connections']} connections")
            print(f"   Bandwidth: ↑{stats['net_up']:.1f}MB/s ↓{stats['net_down']:.1f}MB/s")
            
            recommendations = self.resource_monitor.get_recommendations(self.max_workers)
            print(f"\n💡 System Analysis:")
            for rec in recommendations:
                print(f"   {rec}")
                
            # Add performance diagnostics
            perf_analysis = self.performance_diagnostics.get_bottleneck_analysis()
            print(f"\n{perf_analysis}")
    
    def display_live_stats(self):
        """Display live performance statistics that update in place"""
        last_update = 0
        while self.resource_monitor.monitoring:
            try:
                current_time = time.time()
                # Update every 5 seconds
                if current_time - last_update < 5.0:
                    time.sleep(1.0)
                    continue
                    
                last_update = current_time
                
                # Get current stats
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                # Build stats line
                if self.performance_diagnostics.stats['response_time']:
                    avg_time = sum(self.performance_diagnostics.stats['response_time'][-50:]) / min(50, len(self.performance_diagnostics.stats['response_time']))
                    success_rate = self.performance_diagnostics.stats['successful_requests'] / (self.performance_diagnostics.stats['successful_requests'] + self.performance_diagnostics.stats['failed_requests']) * 100 if (self.performance_diagnostics.stats['successful_requests'] + self.performance_diagnostics.stats['failed_requests']) > 0 else 0
                    stats_line = f"📊 Stats: {self.processed_count} processed | {success_rate:.1f}% success | {avg_time:.1f}s avg | {self.max_workers} workers | CPU {cpu_percent:.1f}% | DNS cache: {len(DNS_CACHE)}"
                else:
                    stats_line = f"📊 Stats: {self.processed_count} processed | {self.max_workers} workers | CPU {cpu_percent:.1f}% | DNS cache: {len(DNS_CACHE)}"
                
                # Update stats in place (above progress bars)
                print(f"\r{' ' * 200}\r{stats_line}", end='', flush=True)
                
                time.sleep(1.0)
                
            except Exception:
                time.sleep(1.0)

    async def process_all_domains(self):
        """Process all domains with optimized concurrent processing and live monitoring"""
        try:
            # Initialize shared HTTP client
            await self.initialize_client()
            
            # Load domains
            print("📂 Loading domains from parquet file...")
            df = pd.read_parquet(self.input_file)
            
            # Auto-detect domain column
            domain_column = None
            times_column = None
            possible_domain_columns = ['domain', 'hplt_domain', 'oscar_domain', 'url', 'website', 'site']
            
            for col in possible_domain_columns:
                if col in df.columns:
                    domain_column = col
                    break
            
            if domain_column is None:
                # If no standard column found, use the first column that looks like domains
                for col in df.columns:
                    if df[col].dtype == 'object':  # String column
                        # Check if it contains domain-like strings
                        sample_values = df[col].dropna().head(10).tolist()
                        if any('.' in str(val) and len(str(val)) > 3 for val in sample_values):
                            domain_column = col
                            break
            
            # Always use the second column as times column if it exists
            if len(df.columns) >= 2:
                times_column = df.columns[1]
            
            if domain_column is None:
                print("❌ Error: Could not find a domain column in the parquet file")
                print(f"Available columns: {df.columns.tolist()}")
                return
            
            print(f"✅ Using column '{domain_column}' for domains")
            if times_column:
                print(f"✅ Using column '{times_column}' for times data")
            
            # Create domain-times mapping using the second column
            domain_times_map = {}
            if times_column:
                for _, row in df.iterrows():
                    domain_times_map[row[domain_column]] = row[times_column]
            
            all_domains = df[domain_column].dropna().tolist()
            total_domains = len(all_domains)
            print(f"📊 Found {total_domains:,} domains to process")
            
            # Load existing results if available (chunk-based resume)
            existing_results = []
            chunk_size = 10000
            start_chunk = 0
            
            if os.path.exists(self.output_file):
                print("📋 Loading existing results...")
                try:
                    existing_df = pd.read_parquet(self.output_file)
                    existing_results = existing_df.to_dict('records')
                    num_existing = len(existing_results)
                    
                    # Calculate which complete chunks we have
                    complete_chunks = num_existing // chunk_size
                    start_chunk = complete_chunks
                    
                    # Only keep results from complete chunks
                    existing_results = existing_results[:complete_chunks * chunk_size]
                    
                    # Calculate remaining domains starting from incomplete chunk
                    start_index = complete_chunks * chunk_size
                    remaining_df = df.iloc[start_index:]
                    
                    print(f"✅ Found {num_existing:,} existing results")
                    print(f"📦 {complete_chunks} complete chunks ({len(existing_results):,} results kept)")
                    if num_existing > len(existing_results):
                        incomplete_domains = num_existing - len(existing_results)
                        print(f"🔄 Discarding {incomplete_domains} results from incomplete chunk")
                    print(f"🚀 Resuming from chunk {start_chunk + 1}, processing {len(remaining_df):,} domains")
                    df = remaining_df  # Use remaining dataframe
                except Exception as e:
                    print(f"⚠️ Error loading existing results: {e}")
                    print("Starting fresh...")
            
            if len(df) == 0:
                print("🎉 All domains already processed!")
                return
            
            # Start resource monitoring in background (no live updates)
            self.resource_monitor.start_monitoring(live_updates=False)
            
            print(f"🚀 Processing {len(df):,} domains with {self.max_workers} workers (all optimizations active)")
            print()  # Space for stats line
            
            # Start live stats display in background
            stats_thread = threading.Thread(target=self.display_live_stats, daemon=True)
            stats_thread.start()
            
            # Process domains in chunks
            all_results = existing_results.copy()
            total_processed = 0  # Track total processed for overall progress
            
            # Create main progress bar (position 1, after stats line)
            # Show total progress including already completed domains
            total_original_domains = total_domains
            already_completed = len(existing_results)
            
            main_pbar = tqdm(
                total=total_original_domains,
                initial=already_completed,
                desc="Overall Progress",
                position=1,
                unit="domain",
                mininterval=0.1,
                maxinterval=0.1,
                leave=True
            )
            
            for chunk_idx in range(0, len(df), chunk_size):
                chunk_df = df.iloc[chunk_idx:chunk_idx + chunk_size]
                chunk_num = chunk_idx//chunk_size + 1
                total_chunks = (len(df)-1)//chunk_size + 1
                
                # Create domain-times pairs for this chunk
                chunk_pairs = []
                for _, row in chunk_df.iterrows():
                    domain = row[domain_column]
                    domain_times = row[times_column] if times_column else None
                    chunk_pairs.append((domain, domain_times))
                
                # Process chunk concurrently with optimizations
                chunk_results = await self.process_domain_pairs_concurrent(chunk_pairs, chunk_idx, main_pbar)
                
                # Add results to main list (no retries)
                all_results.extend(chunk_results)
                
                # Calculate chunk summary BEFORE clearing results
                successful_in_chunk = sum(1 for r in chunk_results if r['title'] or r['meta_description'])
                chunk_size_processed = len(chunk_results)
                
                # Save progress every chunk
                self.save_progress(all_results)
                
                # CRITICAL: Clear chunk_results to free memory immediately
                chunk_results.clear()
                del chunk_results
                
                # Reset resources after each chunk to prevent slowdown
                await self.reset_chunk_resources()
                
                # Force garbage collection
                gc.collect()
                
                # Show chunk summary only for larger chunks or every 5th chunk
                if len(chunk_df) >= 1000 or chunk_num % 5 == 0:
                    print(f"\n✅ Chunk {chunk_num}: {successful_in_chunk}/{chunk_size_processed} successful ({successful_in_chunk/chunk_size_processed*100:.1f}%)")
            
            main_pbar.close()
            
            # Final statistics (no final save needed - already saved after each chunk)
            successful_total = sum(1 for r in all_results if r['title'] or r['meta_description'])
            print(f"\n🎉 Processing completed!")
            print(f"📊 Total processed: {len(all_results):,}")
            print(f"✅ Successful: {successful_total:,} ({successful_total/len(all_results)*100:.1f}%)")
            print(f"💾 Results saved to: {self.output_file}")
            
        finally:
            # Cleanup
            await self.cleanup_client()
            self.resource_monitor.stop_monitoring()

    async def process_domain_pairs_concurrent(self, domain_pairs: List[tuple], chunk_start: int, main_pbar=None):
        """Process domain-times pairs concurrently while maintaining original order"""
        semaphore = asyncio.Semaphore(self.max_workers)
        
        # Track performance for monitoring
        chunk_start_time = time.time()
        successful_in_chunk = 0
        
        # Create progress bar for this chunk
        chunk_pbar = tqdm(
            total=len(domain_pairs),
            desc=f"Chunk Progress",
            position=2,
            leave=False,
            unit="domain",
            mininterval=1,
            maxinterval=1
        )
        
        async def process_single_domain_pair_with_index(index: int, domain_pair: tuple):
            nonlocal successful_in_chunk
            async with semaphore:
                domain, domain_times = domain_pair
                # Call scrape_domain with the exact domain_times for this domain
                result = await self.scrape_domain_with_times(domain, domain_times)
                
                # Track chunk performance
                if result['title'] is not None or result['meta_description'] is not None:
                    successful_in_chunk += 1
                
                # Update both progress bars immediately
                chunk_pbar.update(1)
                if main_pbar:
                    main_pbar.update(1)
                
                # Update chunk progress bar postfix
                chunk_pbar.set_postfix({
                    'Success': f'{successful_in_chunk}/{chunk_pbar.n}',
                    'Workers': self.max_workers,
                    'Delay': f'{self.request_delay:.2f}s'
                })
                
                # Update overall progress bar postfix
                if main_pbar:
                    elapsed = time.time() - self.start_time
                    speed = main_pbar.n / elapsed if elapsed > 0 else 0
                    main_pbar.set_postfix({
                        'Speed': f'{speed:.1f} dom/s',
                        'Workers': self.max_workers,
                        'Success': f'{successful_in_chunk}/{chunk_pbar.n}'
                    })
                
                return index, result
        
        # Create tasks for all domain pairs with their indices
        tasks = [process_single_domain_pair_with_index(i, pair) for i, pair in enumerate(domain_pairs)]
        
        # Store results with their original indices
        indexed_results = []
        
        # Process with progress tracking
        for completed_task in asyncio.as_completed(tasks):
            index, result = await completed_task
            indexed_results.append((index, result))
            self.processed_count += 1
        
        chunk_pbar.close()
        
        # Sort results by original index to maintain order
        indexed_results.sort(key=lambda x: x[0])
        
        # Extract just the results in original order
        results = [result for index, result in indexed_results]
        
        return results

    async def scrape_domain_with_times(self, domain: str, domain_times) -> Dict[str, Optional[str]]:
        """Scrape domain with exact domain_times value (no dictionary lookup)"""
        start_time = time.time()
        error_type = None
        status_code = None

        # Store the original domain and times exactly as they appear in input
        original_domain = domain  # Keep exactly as is from parquet

        try:
            # Clean the domain for the actual HTTP request (but keep original for output)
            clean_domain = domain.strip()
            if clean_domain.startswith('@'):
                clean_domain = clean_domain[1:].strip()
            
            # Ensure clean domain has protocol
            if not clean_domain.startswith(('http://', 'https://')):
                clean_domain = f'https://{clean_domain}'
            
            # DNS caching optimization
            hostname = urlparse(clean_domain).netloc
            if hostname in DNS_CACHE:
                self.performance_diagnostics.record_dns_cache(True)
            else:
                self.performance_diagnostics.record_dns_cache(False)
                # Pre-resolve DNS to cache it
                await resolve_domain_cached(clean_domain)
            
            # Rate limiting mitigation - random delay
            if self.request_delay > 0:
                delay = self.request_delay + random.uniform(0, self.request_delay)
                await asyncio.sleep(delay)
            
            # Rotate user agent for each request
            headers = {'User-Agent': random.choice(self.user_agents)}
            
            # Use shared client with shorter timeout
            try:
                # First try with HTTPS
                try:
                    response = await self.shared_client.get(
                        clean_domain, 
                        timeout=10.0,
                        headers=headers,
                        follow_redirects=True  # Ensure redirects are followed
                    )
                except Exception as e:
                    # If HTTPS fails, try HTTP
                    if clean_domain.startswith('https://'):
                        http_domain = clean_domain.replace('https://', 'http://')
                        response = await self.shared_client.get(
                            http_domain,
                            timeout=10.0,
                            headers=headers,
                            follow_redirects=True
                        )
                    else:
                        raise  # Re-raise if it wasn't an HTTPS issue

                status_code = response.status_code
                final_url = str(response.url)

                # If we got redirected, record the redirect chain
                redirect_url = None
                if response.history:
                    # Record the redirect chain status codes
                    for r in response.history:
                        self.performance_diagnostics.record_request(clean_domain, start_time, time.time(), r.status_code)
                    
                    # Store the final URL as redirect_url if different from clean domain
                    if final_url != clean_domain:
                        redirect_url = final_url

                # Get content with encoding handling
                try:
                    content = response.content[:1000000]  # First try binary content
                    if not content:
                        content = response.text.encode('utf-8')[:1000000]  # Try text content
                except Exception:
                    content = b''  # Empty content as last resort

                self._last_status_code = status_code  # Track for adaptive scaling
                
                # Handle rate limiting first (before raise_for_status)
                if status_code == 429:
                    # Increase delay for future requests
                    self.request_delay = min(self.request_delay * 1.5, 2.0)  # Max 2 seconds
                    self.performance_diagnostics.record_request(clean_domain, start_time, time.time(), status_code)
                    result = {
                        'domain': original_domain,  # Use original domain from parquet
                        'redirect_url': redirect_url,
                        'status_code': status_code,
                        'title': None,
                        'meta_description': None,
                        'keywords': None,
                        'og_title': None,
                        'og_description': None,
                        'og_type': None
                    }
                    if domain_times is not None:
                        result['domain_times'] = domain_times
                    return result

                # Handle successful responses (200-299 range)
                if 200 <= status_code < 300:
                    # Reduce delay on success
                    self.request_delay = max(self.request_delay * 0.95, 0.05)  # Min 50ms
                    
                    # Parse HTML with timeout protection - using proven scraping technique
                    try:
                        # Try to parse with BeautifulSoup, handling encoding issues gracefully
                        try:
                            soup = BeautifulSoup(content, 'html.parser')
                        except Exception:
                            # Fallback: try with response.text if content parsing fails
                            soup = BeautifulSoup(response.text[:1000000], 'html.parser')
                        
                        # Extract title - using proven method from tester (.get_text() works better than .string)
                        title = None
                        title_tag = soup.find('title')
                        if title_tag:
                            title = title_tag.get_text().strip()[:500]  # Limit title length
                        
                        # Extract meta description and other tags
                        meta_description = None
                        keywords = None
                        og_title = None
                        og_description = None
                        og_type = None
                        
                        try:
                            # Get meta description
                            meta_tag = soup.find('meta', attrs={'name': 'description'})
                            if meta_tag and hasattr(meta_tag, 'get'):
                                content = meta_tag.get('content', '')
                                if content:
                                    meta_description = str(content).strip()[:1000]  # Limit description length
                            
                            # Get meta keywords
                            keywords_tag = soup.find('meta', attrs={'name': 'keywords'})
                            if keywords_tag and hasattr(keywords_tag, 'get'):
                                content = keywords_tag.get('content', '')
                                if content:
                                    keywords = str(content).strip()[:1000]  # Limit keywords length
                            
                            # Get specific Open Graph tags
                            og_title_tag = soup.find('meta', attrs={'property': 'og:title'})
                            if og_title_tag and hasattr(og_title_tag, 'get'):
                                content = og_title_tag.get('content', '')
                                if content:
                                    og_title = str(content).strip()[:500]

                            og_desc_tag = soup.find('meta', attrs={'property': 'og:description'})
                            if og_desc_tag and hasattr(og_desc_tag, 'get'):
                                content = og_desc_tag.get('content', '')
                                if content:
                                    og_description = str(content).strip()[:1000]

                            og_type_tag = soup.find('meta', attrs={'property': 'og:type'})
                            if og_type_tag and hasattr(og_type_tag, 'get'):
                                content = og_type_tag.get('content', '')
                                if content:
                                    og_type = str(content).strip()[:100]
                            
                            # Fallback to og:description if no standard meta description found
                            if not meta_description and og_description:
                                meta_description = og_description
                                
                        except Exception as e:
                            # Log meta tag parsing error but continue
                            pass
                        
                        # Record successful request
                        end_time = time.time()
                        self.performance_diagnostics.record_request(clean_domain, start_time, end_time, status_code)
                        
                        result = {
                            'domain': original_domain,  # Use original domain from parquet
                            'redirect_url': redirect_url,
                            'status_code': status_code,
                            'title': title,
                            'meta_description': meta_description,
                            'keywords': keywords,
                            'og_title': og_title,
                            'og_description': og_description,
                            'og_type': og_type
                        }
                        if domain_times is not None:
                            result['domain_times'] = domain_times
                        
                        # Debug logging for empty results (uncomment if needed for debugging)
                        if not title and not meta_description:
                            print(f"DEBUG: Empty result for {original_domain} - HTML size: {len(content)} bytes - Status: {status_code}")
                        
                        return result
                        
                    except Exception as e:
                        # HTML parsing failed but we got the response - log the error
                        print(f"DEBUG: HTML parsing failed for {original_domain}: {type(e).__name__}: {str(e)}")
                        end_time = time.time()
                        self.performance_diagnostics.record_request(clean_domain, start_time, end_time, status_code)
                else:
                    # Record non-200 status
                    end_time = time.time()
                    self.performance_diagnostics.record_request(clean_domain, start_time, end_time, status_code)
                
            except asyncio.TimeoutError:
                error_type = "Timeout"
                status_code = None
            except httpx.HTTPStatusError as e:
                error_type = f"HTTP_{e.response.status_code}"
                status_code = e.response.status_code if hasattr(e, 'response') else None
            except httpx.ConnectError:
                error_type = "Connection"
                status_code = None
            except httpx.UnsupportedProtocol:
                error_type = "Protocol"
                status_code = None
            except httpx.InvalidURL:
                error_type = "InvalidURL"
                status_code = None
            except httpx.RequestError:
                error_type = "Request"
                status_code = None
            except Exception as e:
                error_type = f"Other: {type(e).__name__}"
                status_code = None
                
        except asyncio.TimeoutError:
            error_type = "Timeout"
            status_code = None
        except httpx.HTTPStatusError as e:
            error_type = f"HTTP_{e.response.status_code}"
            status_code = e.response.status_code if hasattr(e, 'response') else None
        except httpx.ConnectError:
            error_type = "Connection"
            status_code = None
        except httpx.UnsupportedProtocol:
            error_type = "Protocol"
            status_code = None
        except httpx.InvalidURL:
            error_type = "InvalidURL"
            status_code = None
        except httpx.RequestError:
            error_type = "Request"
            status_code = None
        except Exception as e:
            error_type = f"Other: {type(e).__name__}"
            status_code = None
            
        # Record failed request
        end_time = time.time()
        self.performance_diagnostics.record_request(clean_domain, start_time, end_time, error_type=error_type)
        
        result = {
            'domain': original_domain,  # Use original domain from parquet
            'redirect_url': None,  # No redirect for errors
            'status_code': status_code,  # Include status code even for errors
            'title': None,
            'meta_description': None,
            'keywords': None,
            'og_title': None,
            'og_description': None,
            'og_type': None
        }
        if domain_times is not None:
            result['domain_times'] = domain_times
        return result

async def main():
    if len(sys.argv) < 2:
        print("Usage: python3 sequential_domain_scraper.py <input_parquet_file> [max_workers]")
        print("Example: python3 sequential_domain_scraper.py domains.parquet 15")
        sys.exit(1)
    
    input_file = sys.argv[1]
    max_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    if not input_file.endswith('.parquet'):
        print("Error: Input file must be a .parquet file")
        sys.exit(1)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    scraper = DomainScraper(input_file, max_workers)
    await scraper.process_all_domains()

if __name__ == "__main__":
    asyncio.run(main()) 