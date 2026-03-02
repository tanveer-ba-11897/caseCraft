"""
Web Loader for CaseCraft Knowledge Base.

Supports:
- Sitemap.xml crawling
- Single URL loading
- URL list loading
- Clean HTML to text extraction
"""

import logging
import time
import ipaddress
import socket
from defusedxml.ElementTree import fromstring as xml_fromstring
from typing import List, Optional, Set  # Optional used in CrawlConfig
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

from core.knowledge.models import RawDocument

logger = logging.getLogger("casecraft.web_loader")


# ─── Security: URL Validation (SSRF Prevention) ─────────────────────────────
BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def validate_url(url: str) -> str:
    """
    Validate that a URL is safe to fetch.
    Blocks private/internal IPs and non-HTTP schemes to prevent SSRF attacks.
    Also resolves DNS to prevent DNS rebinding attacks.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: '{parsed.scheme}'. Only http/https allowed.")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname.")

    if hostname in BLOCKED_HOSTS:
        raise ValueError(f"Blocked host: {hostname}")

    # Block private IP ranges (10.x, 172.16-31.x, 192.168.x, 169.254.x)
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError(f"Blocked private/internal IP: {hostname}")
    except ValueError as e:
        if "Blocked" in str(e):
            raise
        # hostname is a domain name, not an IP — resolve it to check
        try:
            resolved_ip = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(resolved_ip)
            if ip.is_private or ip.is_loopback or ip.is_link_local:
                raise ValueError(f"Blocked: {hostname} resolves to private IP {resolved_ip}")
        except socket.gaierror:
            raise ValueError(f"DNS resolution failed for: {hostname}")

    return url


@dataclass
class CrawlConfig:
    """Configuration for web crawling."""
    delay_between_requests: float = 1.0  # seconds
    timeout: int = 30  # seconds
    max_pages: int = 500  # safety limit
    user_agent: str = "CaseCraft/1.0 (Documentation Crawler)"
    exclude_patterns: Optional[List[str]] = None  # URL patterns to skip
    
    def __post_init__(self):
        if self.exclude_patterns is None:
            self.exclude_patterns = []


class WebLoaderError(Exception):
    """Raised when web loading fails."""
    pass


def _fetch_url(url: str, config: CrawlConfig) -> str:
    """Fetch URL content with proper headers, timeout, and SSRF protection."""
    validate_url(url)

    headers = {
        "User-Agent": config.user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    
    try:
        response = requests.get(
            url, headers=headers, timeout=config.timeout, allow_redirects=False
        )
        # Manually follow redirects with SSRF validation on each target
        redirect_count = 0
        while response.is_redirect and redirect_count < 5:
            redirect_url = urljoin(url, response.headers.get("Location", ""))
            validate_url(redirect_url)
            url = redirect_url
            response = requests.get(
                url, headers=headers, timeout=config.timeout, allow_redirects=False
            )
            redirect_count += 1
        if response.is_redirect:
            raise WebLoaderError(f"Too many redirects for {url}")
        response.raise_for_status()
        return response.text
    except ValueError as e:
        raise WebLoaderError(f"URL validation failed: {e}") from e
    except requests.RequestException as e:
        raise WebLoaderError(f"Failed to fetch {url}: {e}") from e


def _extract_text_from_html(html: str, url: str) -> str:
    """
    Extract clean text from HTML, removing navigation, scripts, etc.
    Focuses on main content areas.
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove unwanted elements
    for element in soup.find_all([
        'script', 'style', 'nav', 'header', 'footer', 
        'aside', 'form', 'button', 'iframe', 'noscript'
    ]):
        element.decompose()
    
    # Remove common navigation/sidebar classes
    for selector in [
        '.nav', '.navbar', '.sidebar', '.menu', '.footer', '.header',
        '.navigation', '.breadcrumb', '.pagination', '.toc', '.table-of-contents',
        '[role="navigation"]', '[role="banner"]', '[role="contentinfo"]'
    ]:
        for element in soup.select(selector):
            element.decompose()
    
    # Try to find main content area
    main_content = None
    for selector in ['main', 'article', '.content', '.main-content', '#content', '#main']:
        main_content = soup.select_one(selector)
        if main_content:
            break
    
    # Use main content if found, otherwise use body
    if main_content:
        text = main_content.get_text(separator='\n', strip=True)
    else:
        body = soup.find('body')
        if body:
            text = body.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
    
    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    text = '\n'.join(lines)
    
    # Skip if too short (likely error page or empty)
    if len(text) < 100:
        return ""
    
    return text


def _parse_sitemap(sitemap_url: str, config: CrawlConfig) -> List[str]:
    """
    Parse a sitemap.xml and extract all page URLs.
    Handles both standard sitemaps and sitemap indexes.
    """
    urls = []
    
    try:
        content = _fetch_url(sitemap_url, config)
    except WebLoaderError as e:
        logger.warning("Could not fetch sitemap: %s", e)
        return urls
    
    try:
        root = xml_fromstring(content)
    except Exception as e:
        raise WebLoaderError(f"Invalid sitemap XML: {e}") from e
    
    # Handle namespace
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}")[0] + "}"
    
    # Check if this is a sitemap index
    sitemap_tags = root.findall(f".//{namespace}sitemap")
    if sitemap_tags:
        logger.info("Found sitemap index with %d sitemaps", len(sitemap_tags))
        for sitemap in sitemap_tags:
            loc = sitemap.find(f"{namespace}loc")
            if loc is not None and loc.text:
                # Recursively parse child sitemaps
                child_urls = _parse_sitemap(loc.text.strip(), config)
                urls.extend(child_urls)
                time.sleep(config.delay_between_requests)
    else:
        # Standard sitemap with URL entries
        url_tags = root.findall(f".//{namespace}url")
        for url_tag in url_tags:
            loc = url_tag.find(f"{namespace}loc")
            if loc is not None and loc.text:
                urls.append(loc.text.strip())
    
    return urls


def _should_exclude(url: str, patterns: List[str]) -> bool:
    """Check if URL matches any exclusion pattern."""
    url_lower = url.lower()
    for pattern in patterns:
        if pattern.lower() in url_lower:
            return True
    return False


def load_from_sitemap(
    sitemap_url: str,
    config: Optional[CrawlConfig] = None,
) -> List[RawDocument]:
    """
    Crawl a sitemap and load all pages as RawDocuments.
    
    Args:
        sitemap_url: URL to sitemap.xml
        config: Crawl configuration (optional)
        
    Returns:
        List of RawDocument objects containing page content
    """
    if config is None:
        config = CrawlConfig()
    
    logger.info("Parsing sitemap: %s", sitemap_url)
    urls = _parse_sitemap(sitemap_url, config)
    
    if not urls:
        logger.warning("No URLs found in sitemap")
        return []
    
    # Filter and limit URLs
    filtered_urls = []
    for url in urls:
        if _should_exclude(url, config.exclude_patterns):
            continue
        filtered_urls.append(url)
        if len(filtered_urls) >= config.max_pages:
            logger.warning("Reached max_pages limit (%d)", config.max_pages)
            break
    
    logger.info("Found %d pages to crawl", len(filtered_urls))
    
    documents = []
    for i, url in enumerate(filtered_urls):
        logger.info("Crawling [%d/%d]: %s", i+1, len(filtered_urls), url)
        
        try:
            html = _fetch_url(url, config)
            text = _extract_text_from_html(html, url)
            
            if text:
                # Extract page title for source name
                soup = BeautifulSoup(html, 'html.parser')
                title = soup.find('title')
                source_name = title.get_text(strip=True) if title else url
                
                documents.append(RawDocument(
                    text=text,
                    source_name=source_name[:100],  # Limit length
                    source_type="web_doc",
                ))
                logger.info("Extracted %d chars", len(text))
            else:
                logger.info("No content extracted (skipped)")
                
        except WebLoaderError as e:
            logger.error("%s", e)
            continue
        
        # Rate limiting
        if i < len(filtered_urls) - 1:
            time.sleep(config.delay_between_requests)
    
    logger.info("Successfully loaded %d documents", len(documents))
    return documents


def load_from_url(url: str, config: Optional[CrawlConfig] = None) -> RawDocument:
    """
    Load a single URL as a RawDocument.
    
    Args:
        url: URL to load
        config: Crawl configuration (optional)
        
    Returns:
        RawDocument containing page content
    """
    if config is None:
        config = CrawlConfig()
    
    html = _fetch_url(url, config)
    text = _extract_text_from_html(html, url)
    
    if not text:
        raise WebLoaderError(f"No content extracted from {url}")
    
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.find('title')
    source_name = title.get_text(strip=True) if title else url
    
    return RawDocument(
        text=text,
        source_name=source_name[:100],
        source_type="web_doc",
    )


def load_from_url_list(
    urls: List[str],
    config: Optional[CrawlConfig] = None,
) -> List[RawDocument]:
    """
    Load multiple URLs as RawDocuments.
    
    Args:
        urls: List of URLs to load
        config: Crawl configuration (optional)
        
    Returns:
        List of RawDocument objects
    """
    if config is None:
        config = CrawlConfig()
    
    documents = []
    for i, url in enumerate(urls):
        logger.info("Loading [%d/%d]: %s", i+1, len(urls), url)
        
        try:
            doc = load_from_url(url, config)
            documents.append(doc)
            logger.info("Extracted %d chars", len(doc.text))
        except WebLoaderError as e:
            logger.error("%s", e)
            continue
        
        if i < len(urls) - 1:
            time.sleep(config.delay_between_requests)
    
    return documents
