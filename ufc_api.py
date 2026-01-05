#!/usr/bin/env python3
"""
UFC Stats API
FastAPI wrapper for serving scraped UFC data
"""

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from cachetools import TTLCache
from dotenv import load_dotenv
from typing import Optional
from collections import defaultdict
import json
import os
import httpx
import asyncio
from datetime import datetime, date

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# CONFIGURATION
# =============================================================================

# Rate limit: 100 requests per day, 100 per minute burst protection
RATE_LIMIT = "100/day;100/minute"

# Analytics secret key (loaded from .env file)
ANALYTICS_SECRET = os.getenv("ANALYTICS_SECRET")

# Treblle configuration
TREBLLE_API_KEY = os.getenv("TREBLLE_API_KEY")
TREBLLE_SDK_TOKEN = os.getenv("TREBLLE_SDK_TOKEN")
TREBLLE_API_URL = "https://rocknrolla.treblle.com"

# Cache settings (5 minute TTL, max 1000 items)
cache = TTLCache(maxsize=1000, ttl=300)

# Rate limiter by IP
limiter = Limiter(key_func=get_remote_address)


# =============================================================================
# ANALYTICS
# =============================================================================

ANALYTICS_FILE = os.path.join(os.path.dirname(__file__), "analytics.json")

def parse_user_agent(ua_string: str) -> dict:
    """Parse user agent to extract device, browser, and OS info"""
    ua = ua_string.lower() if ua_string else ""

    # Device type
    if any(x in ua for x in ["mobile", "android", "iphone", "ipad"]):
        if "ipad" in ua or "tablet" in ua:
            device = "Tablet"
        else:
            device = "Mobile"
    elif "bot" in ua or "crawler" in ua or "spider" in ua:
        device = "Bot"
    else:
        device = "Desktop"

    # Browser
    if "firefox" in ua:
        browser = "Firefox"
    elif "edg" in ua:
        browser = "Edge"
    elif "chrome" in ua:
        browser = "Chrome"
    elif "safari" in ua:
        browser = "Safari"
    elif "opera" in ua:
        browser = "Opera"
    elif "curl" in ua:
        browser = "cURL"
    elif "python" in ua:
        browser = "Python"
    elif "postman" in ua:
        browser = "Postman"
    else:
        browser = "Other"

    # OS
    if "windows" in ua:
        os_name = "Windows"
    elif "mac" in ua or "darwin" in ua:
        os_name = "macOS"
    elif "linux" in ua:
        os_name = "Linux"
    elif "android" in ua:
        os_name = "Android"
    elif "iphone" in ua or "ipad" in ua or "ios" in ua:
        os_name = "iOS"
    else:
        os_name = "Other"

    return {"device": device, "browser": browser, "os": os_name}

# In-memory analytics (saved to file periodically)
analytics = {
    "total_requests": 0,
    "requests_today": 0,
    "today": str(date.today()),
    "endpoints": defaultdict(int),
    "ips": defaultdict(int),
    "daily": defaultdict(int),
    "hourly": defaultdict(int),
    "devices": defaultdict(int),
    "browsers": defaultdict(int),
    "os": defaultdict(int),
    "referers": defaultdict(int),
    "methods": defaultdict(int),
    "status_codes": defaultdict(int),
    "user_journeys": defaultdict(list)  # IP -> list of {timestamp, endpoint, method}
}

def load_analytics():
    """Load analytics from file"""
    global analytics
    if os.path.exists(ANALYTICS_FILE):
        try:
            with open(ANALYTICS_FILE) as f:
                data = json.load(f)
                analytics["total_requests"] = data.get("total_requests", 0)
                analytics["requests_today"] = data.get("requests_today", 0)
                analytics["today"] = data.get("today", str(date.today()))
                analytics["endpoints"] = defaultdict(int, data.get("endpoints", {}))
                analytics["ips"] = defaultdict(int, data.get("ips", {}))
                analytics["daily"] = defaultdict(int, data.get("daily", {}))
                analytics["hourly"] = defaultdict(int, data.get("hourly", {}))
                analytics["devices"] = defaultdict(int, data.get("devices", {}))
                analytics["browsers"] = defaultdict(int, data.get("browsers", {}))
                analytics["os"] = defaultdict(int, data.get("os", {}))
                analytics["referers"] = defaultdict(int, data.get("referers", {}))
                analytics["methods"] = defaultdict(int, data.get("methods", {}))
                analytics["status_codes"] = defaultdict(int, data.get("status_codes", {}))
                analytics["user_journeys"] = defaultdict(list, data.get("user_journeys", {}))
        except:
            pass

def save_analytics():
    """Save analytics to file - called on every request for persistence"""
    try:
        with open(ANALYTICS_FILE, "w") as f:
            json.dump({
                "total_requests": analytics["total_requests"],
                "requests_today": analytics["requests_today"],
                "today": analytics["today"],
                "endpoints": dict(analytics["endpoints"]),
                "ips": dict(analytics["ips"]),
                "daily": dict(analytics["daily"]),
                "hourly": dict(analytics["hourly"]),
                "devices": dict(analytics["devices"]),
                "browsers": dict(analytics["browsers"]),
                "os": dict(analytics["os"]),
                "referers": dict(analytics["referers"]),
                "methods": dict(analytics["methods"]),
                "status_codes": dict(analytics["status_codes"]),
                "user_journeys": dict(analytics["user_journeys"])
            }, f, indent=2)
    except:
        pass

def track_request(request: Request):
    """Track an API request with detailed metrics"""
    global analytics

    # Reset daily counter if new day
    today = str(date.today())
    if analytics["today"] != today:
        analytics["requests_today"] = 0
        analytics["today"] = today

    # Basic metrics
    analytics["total_requests"] += 1
    analytics["requests_today"] += 1
    analytics["endpoints"][request.url.path] += 1
    analytics["ips"][get_remote_address(request)] += 1
    analytics["daily"][today] += 1
    analytics["methods"][request.method] += 1

    # Time-based tracking
    current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
    analytics["hourly"][current_hour] += 1

    # User agent parsing
    user_agent = request.headers.get("user-agent", "")
    ua_info = parse_user_agent(user_agent)
    analytics["devices"][ua_info["device"]] += 1
    analytics["browsers"][ua_info["browser"]] += 1
    analytics["os"][ua_info["os"]] += 1

    # Referer tracking
    referer = request.headers.get("referer", "Direct")
    if referer != "Direct":
        # Extract domain from referer
        try:
            from urllib.parse import urlparse
            referer = urlparse(referer).netloc or "Direct"
        except:
            referer = "Direct"
    analytics["referers"][referer] += 1

    # Track user journey (IP -> list of actions with timestamps and full details)
    ip = get_remote_address(request)
    journey_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "endpoint": request.url.path,
        "method": request.method,
        "referer": referer,
        "device": ua_info["device"],
        "browser": ua_info["browser"],
        "os": ua_info["os"]
    }
    analytics["user_journeys"][ip].append(journey_entry)
    # Keep only last 500 entries per IP to allow for date range analysis
    if len(analytics["user_journeys"][ip]) > 500:
        analytics["user_journeys"][ip] = analytics["user_journeys"][ip][-500:]

    # Save on every request for persistence
    save_analytics()

# Load existing analytics on startup
load_analytics()


# =============================================================================
# CACHING HELPERS
# =============================================================================

def get_cached(key: str):
    """Get item from cache"""
    return cache.get(key)

def set_cached(key: str, value):
    """Set item in cache"""
    cache[key] = value
    return value


class PrettyJSONResponse(JSONResponse):
    """Returns formatted/indented JSON for easy reading"""
    def render(self, content) -> bytes:
        return json.dumps(content, indent=2, ensure_ascii=False).encode("utf-8")

app = FastAPI(
    title="UFC Stats API",
    description="Free API for UFC fighter stats, events, and fight data. Free tier: 100 requests/day per IP. Contact for higher limits.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    default_response_class=PrettyJSONResponse
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Analytics middleware
@app.middleware("http")
async def analytics_middleware(request: Request, call_next):
    # Track the request (skip analytics endpoint itself)
    if not request.url.path.startswith("/analytics"):
        track_request(request)
    response = await call_next(request)
    return response


# Treblle middleware - sends API analytics to Treblle
async def send_to_treblle(payload: dict):
    """Send request data to Treblle in the background"""
    if not TREBLLE_API_KEY or not TREBLLE_SDK_TOKEN:
        return
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                TREBLLE_API_URL,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": TREBLLE_API_KEY,
                },
                timeout=10.0
            )
    except Exception:
        pass  # Silently fail - don't affect API response

@app.middleware("http")
async def treblle_middleware(request: Request, call_next):
    """Capture request/response data and send to Treblle"""
    if not TREBLLE_API_KEY or not TREBLLE_SDK_TOKEN:
        return await call_next(request)

    # Skip analytics endpoints
    if request.url.path.startswith("/analytics"):
        return await call_next(request)

    start_time = datetime.now()

    # Capture request body
    request_body = None
    try:
        body_bytes = await request.body()
        if body_bytes:
            request_body = json.loads(body_bytes.decode())
    except:
        request_body = None

    # Process request
    response = await call_next(request)

    # Calculate duration
    duration = (datetime.now() - start_time).total_seconds()

    # Capture response body
    response_body = None
    response_body_bytes = b""
    async for chunk in response.body_iterator:
        response_body_bytes += chunk
    try:
        response_body = json.loads(response_body_bytes.decode())
    except:
        response_body = None

    # Rebuild response with captured body
    from starlette.responses import Response
    new_response = Response(
        content=response_body_bytes,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type
    )

    # Build Treblle payload
    treblle_payload = {
        "api_key": TREBLLE_API_KEY,
        "project_id": TREBLLE_SDK_TOKEN,
        "version": 0.6,
        "sdk": "python-fastapi-custom",
        "data": {
            "server": {
                "ip": "127.0.0.1",
                "timezone": "UTC",
                "software": "uvicorn",
                "protocol": request.url.scheme,
            },
            "language": {
                "name": "python",
                "version": "3.10"
            },
            "request": {
                "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "ip": get_remote_address(request),
                "url": str(request.url),
                "user_agent": request.headers.get("user-agent", ""),
                "method": request.method,
                "headers": dict(request.headers),
                "body": request_body
            },
            "response": {
                "headers": dict(new_response.headers),
                "code": new_response.status_code,
                "size": len(response_body_bytes),
                "load_time": duration,
                "body": response_body
            }
        }
    }

    # Send to Treblle in background (don't wait)
    asyncio.create_task(send_to_treblle(treblle_payload))

    return new_response


# Data storage (loaded from JSON files)
DATA_DIR = os.getenv("UFC_DATA_DIR", "ufc_data")
FIGHTERS = []
EVENTS = []
FIGHTS = []
FIGHTERS_BY_ID = {}
EVENTS_BY_ID = {}
FIGHTS_BY_ID = {}


def load_data():
    """Load scraped data from JSON files"""
    global FIGHTERS, EVENTS, FIGHTS, FIGHTERS_BY_ID, EVENTS_BY_ID, FIGHTS_BY_ID

    fighters_path = os.path.join(DATA_DIR, "fighters.json")
    events_path = os.path.join(DATA_DIR, "events.json")
    fights_path = os.path.join(DATA_DIR, "fights.json")

    if os.path.exists(fighters_path):
        with open(fighters_path) as f:
            FIGHTERS = json.load(f)
            FIGHTERS_BY_ID = {f["id"]: f for f in FIGHTERS if f.get("id")}

    if os.path.exists(events_path):
        with open(events_path) as f:
            EVENTS = json.load(f)
            EVENTS_BY_ID = {e["id"]: e for e in EVENTS if e.get("id")}

    if os.path.exists(fights_path):
        with open(fights_path) as f:
            FIGHTS = json.load(f)
            FIGHTS_BY_ID = {f["id"]: f for f in FIGHTS if f.get("id")}

    print(f"Loaded {len(FIGHTERS)} fighters, {len(EVENTS)} events, {len(FIGHTS)} fights")


@app.on_event("startup")
async def startup():
    load_data()


# =============================================================================
# FIGHTER ENDPOINTS
# =============================================================================

@app.get("/api/fighters", tags=["Fighters"])
@limiter.limit(RATE_LIMIT)
async def list_fighters(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    weight_class: Optional[str] = None,
    search: Optional[str] = None,
    min_wins: Optional[int] = None,
    stance: Optional[str] = None,
    sort_by: Optional[str] = Query(None, description="Sort by: wins, losses, slpm, td_avg")
):
    """
    List all fighters with optional filtering and pagination.
    Results are cached for 5 minutes.
    """
    # Check cache
    cache_key = f"fighters:{limit}:{offset}:{search}:{min_wins}:{stance}:{sort_by}"
    cached = get_cached(cache_key)
    if cached:
        return cached

    results = FIGHTERS.copy()

    # Filter by search term
    if search:
        search_lower = search.lower()
        results = [f for f in results if search_lower in (f.get("name", "") or "").lower()]

    # Filter by minimum wins
    if min_wins:
        results = [f for f in results if (f.get("wins") or 0) >= min_wins]

    # Filter by stance
    if stance:
        results = [f for f in results if (f.get("stance") or "").lower() == stance.lower()]

    # Sort
    if sort_by:
        reverse = True  # Default descending
        if sort_by == "losses":
            reverse = True
        results = sorted(results, key=lambda x: x.get(sort_by) or 0, reverse=reverse)

    # Pagination
    total = len(results)
    results = results[offset:offset + limit]

    response = {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": results
    }
    return set_cached(cache_key, response)


# NOTE: /compare must come BEFORE /{fighter_id} to avoid route conflicts
@app.get("/api/fighters/compare", tags=["Fighters"])
@limiter.limit(RATE_LIMIT)
async def compare_fighters(
    request: Request,
    fighter1: str = Query(..., description="First fighter ID"),
    fighter2: str = Query(..., description="Second fighter ID")
):
    """Compare two fighters head-to-head. Rate limit: 100 requests/minute per IP."""
    f1 = FIGHTERS_BY_ID.get(fighter1)
    f2 = FIGHTERS_BY_ID.get(fighter2)
    
    if not f1:
        raise HTTPException(status_code=404, detail=f"Fighter {fighter1} not found")
    if not f2:
        raise HTTPException(status_code=404, detail=f"Fighter {fighter2} not found")
    
    def safe_get(d, key, default=0):
        return d.get(key) if d.get(key) is not None else default
    
    comparison = {
        "fighter1": {
            "id": fighter1,
            "name": f1.get("name"),
            "record": f"{f1.get('wins', 0)}-{f1.get('losses', 0)}-{f1.get('draws', 0)}"
        },
        "fighter2": {
            "id": fighter2,
            "name": f2.get("name"),
            "record": f"{f2.get('wins', 0)}-{f2.get('losses', 0)}-{f2.get('draws', 0)}"
        },
        "stats_comparison": {
            "slpm": {"fighter1": safe_get(f1, "slpm"), "fighter2": safe_get(f2, "slpm")},
            "str_acc": {"fighter1": safe_get(f1, "str_acc"), "fighter2": safe_get(f2, "str_acc")},
            "sapm": {"fighter1": safe_get(f1, "sapm"), "fighter2": safe_get(f2, "sapm")},
            "str_def": {"fighter1": safe_get(f1, "str_def"), "fighter2": safe_get(f2, "str_def")},
            "td_avg": {"fighter1": safe_get(f1, "td_avg"), "fighter2": safe_get(f2, "td_avg")},
            "td_acc": {"fighter1": safe_get(f1, "td_acc"), "fighter2": safe_get(f2, "td_acc")},
            "td_def": {"fighter1": safe_get(f1, "td_def"), "fighter2": safe_get(f2, "td_def")},
            "sub_avg": {"fighter1": safe_get(f1, "sub_avg"), "fighter2": safe_get(f2, "sub_avg")},
        },
        "advantages": {
            "fighter1": [],
            "fighter2": []
        }
    }
    
    # Calculate advantages
    stats_to_compare = [
        ("slpm", "Strikes Landed/Min", True),  # Higher is better
        ("str_acc", "Strike Accuracy", True),
        ("sapm", "Strikes Absorbed/Min", False),  # Lower is better
        ("str_def", "Strike Defense", True),
        ("td_avg", "Takedown Avg", True),
        ("td_acc", "Takedown Accuracy", True),
        ("td_def", "Takedown Defense", True),
        ("sub_avg", "Submission Avg", True),
    ]
    
    for stat, label, higher_is_better in stats_to_compare:
        v1 = safe_get(f1, stat)
        v2 = safe_get(f2, stat)
        if v1 and v2:
            if higher_is_better:
                if v1 > v2:
                    comparison["advantages"]["fighter1"].append(label)
                elif v2 > v1:
                    comparison["advantages"]["fighter2"].append(label)
            else:
                if v1 < v2:
                    comparison["advantages"]["fighter1"].append(label)
                elif v2 < v1:
                    comparison["advantages"]["fighter2"].append(label)
    
    return comparison


# Dynamic routes must come AFTER /compare
@app.get("/api/fighters/{fighter_id}", tags=["Fighters"])
@limiter.limit(RATE_LIMIT)
async def get_fighter(request: Request, fighter_id: str):
    """Get detailed information for a specific fighter. Cached for 5 minutes."""
    cache_key = f"fighter:{fighter_id}"
    cached = get_cached(cache_key)
    if cached:
        return cached

    fighter = FIGHTERS_BY_ID.get(fighter_id)
    if not fighter:
        raise HTTPException(status_code=404, detail="Fighter not found")
    return set_cached(cache_key, fighter)


@app.get("/api/fighters/{fighter_id}/stats", tags=["Fighters"])
@limiter.limit(RATE_LIMIT)
async def get_fighter_stats(request: Request, fighter_id: str):
    """Get career statistics for a fighter. Rate limit: 100 requests/minute per IP."""
    fighter = FIGHTERS_BY_ID.get(fighter_id)
    if not fighter:
        raise HTTPException(status_code=404, detail="Fighter not found")

    return {
        "id": fighter_id,
        "name": fighter.get("name"),
        "record": {
            "wins": fighter.get("wins", 0),
            "losses": fighter.get("losses", 0),
            "draws": fighter.get("draws", 0),
            "total_fights": fighter.get("wins", 0) + fighter.get("losses", 0) + fighter.get("draws", 0)
        },
        "striking": {
            "slpm": fighter.get("slpm"),
            "str_acc": fighter.get("str_acc"),
            "sapm": fighter.get("sapm"),
            "str_def": fighter.get("str_def")
        },
        "grappling": {
            "td_avg": fighter.get("td_avg"),
            "td_acc": fighter.get("td_acc"),
            "td_def": fighter.get("td_def"),
            "sub_avg": fighter.get("sub_avg")
        },
        "physical": {
            "height": fighter.get("height"),
            "weight": fighter.get("weight"),
            "reach": fighter.get("reach"),
            "stance": fighter.get("stance")
        }
    }


# =============================================================================
# EVENT ENDPOINTS
# =============================================================================

@app.get("/api/events", tags=["Events"])
@limiter.limit(RATE_LIMIT)
async def list_events(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    year: Optional[int] = None,
    search: Optional[str] = None
):
    """List all UFC events with pagination. Rate limit: 100 requests/minute per IP."""
    results = EVENTS.copy()
    
    if search:
        search_lower = search.lower()
        results = [e for e in results if search_lower in (e.get("name", "") or "").lower()]
    
    # Note: Would filter by year if date parsing was implemented
    
    total = len(results)
    results = results[offset:offset + limit]
    
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": results
    }


@app.get("/api/events/{event_id}", tags=["Events"])
@limiter.limit(RATE_LIMIT)
async def get_event(request: Request, event_id: str):
    """Get detailed information for a specific event. Rate limit: 100 requests/minute per IP."""
    event = EVENTS_BY_ID.get(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event


# =============================================================================
# FIGHT ENDPOINTS
# =============================================================================

@app.get("/api/fights", tags=["Fights"])
@limiter.limit(RATE_LIMIT)
async def list_fights(
    request: Request,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    method: Optional[str] = None,
    search: Optional[str] = None
):
    """List all fights with detailed stats. Rate limit: 100 requests/minute per IP."""
    results = FIGHTS.copy()

    if search:
        search_lower = search.lower()
        def fight_matches(fight):
            totals = fight.get("totals") or {}
            f1 = totals.get("fighter1") or {}
            f2 = totals.get("fighter2") or {}
            f1_name = (f1.get("fighter") or "").lower()
            f2_name = (f2.get("fighter") or "").lower()
            return search_lower in f1_name or search_lower in f2_name
        results = [f for f in results if fight_matches(f)]

    if method:
        method_lower = method.lower()
        results = [f for f in results if method_lower in (f.get("method", "") or "").lower()]

    total = len(results)
    results = results[offset:offset + limit]

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": results
    }


@app.get("/api/fights/{fight_id}", tags=["Fights"])
@limiter.limit(RATE_LIMIT)
async def get_fight(request: Request, fight_id: str):
    """Get detailed round-by-round stats for a specific fight. Rate limit: 100 requests/minute per IP."""
    fight = FIGHTS_BY_ID.get(fight_id)
    if not fight:
        raise HTTPException(status_code=404, detail="Fight not found")
    return fight


# =============================================================================
# STATS & ANALYTICS ENDPOINTS
# =============================================================================

@app.get("/api/stats/leaders", tags=["Stats"])
@limiter.limit(RATE_LIMIT)
async def stat_leaders(
    request: Request,
    stat: str = Query("slpm", description="Stat to rank by: slpm, str_acc, td_avg, td_acc, sub_avg, wins"),
    limit: int = Query(10, ge=1, le=50),
    min_fights: int = Query(5, description="Minimum fights to qualify")
):
    """Get stat leaders for a specific statistic. Rate limit: 100 requests/minute per IP."""
    # Filter fighters with minimum fights
    qualified = [
        f for f in FIGHTERS 
        if (f.get("wins", 0) + f.get("losses", 0) + f.get("draws", 0)) >= min_fights
    ]
    
    # Sort by requested stat
    if stat == "wins":
        sorted_fighters = sorted(qualified, key=lambda x: x.get("wins") or 0, reverse=True)
    else:
        sorted_fighters = sorted(
            [f for f in qualified if f.get(stat) is not None],
            key=lambda x: x.get(stat) or 0,
            reverse=True
        )
    
    leaders = []
    for i, f in enumerate(sorted_fighters[:limit]):
        leaders.append({
            "rank": i + 1,
            "id": f.get("id"),
            "name": f.get("name"),
            "value": f.get(stat) if stat != "wins" else f.get("wins"),
            "record": f"{f.get('wins', 0)}-{f.get('losses', 0)}-{f.get('draws', 0)}"
        })
    
    return {
        "stat": stat,
        "min_fights": min_fights,
        "leaders": leaders
    }


@app.get("/api/stats/overview", tags=["Stats"])
@limiter.limit(RATE_LIMIT)
async def stats_overview(request: Request):
    """Get overview statistics about the database. Cached for 5 minutes."""
    cache_key = "stats:overview"
    cached = get_cached(cache_key)
    if cached:
        return cached

    response = {
        "total_fighters": len(FIGHTERS),
        "total_events": len(EVENTS),
        "total_fights": len(FIGHTS),
        "champions": len([f for f in FIGHTERS if f.get("is_champion")]),
        "last_updated": datetime.now().isoformat(),
        "stances": {
            "orthodox": len([f for f in FIGHTERS if (f.get("stance") or "").lower() == "orthodox"]),
            "southpaw": len([f for f in FIGHTERS if (f.get("stance") or "").lower() == "southpaw"]),
            "switch": len([f for f in FIGHTERS if (f.get("stance") or "").lower() == "switch"]),
        }
    }
    return set_cached(cache_key, response)


# =============================================================================
# ANALYTICS ENDPOINT (Protected)
# =============================================================================

@app.get("/analytics", tags=["System"], include_in_schema=False, response_class=HTMLResponse)
async def get_analytics(request: Request, secret: str = Query(..., description="Analytics secret key")):
    """
    View API analytics dashboard (protected by secret key).
    Access: /analytics?secret=YOUR_SECRET_KEY
    """
    if secret != ANALYTICS_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret key")

    # Save current analytics before returning
    save_analytics()

    # Get top endpoints
    top_endpoints = sorted(analytics["endpoints"].items(), key=lambda x: x[1], reverse=True)[:10]

    # Get top IPs (unique users)
    top_ips = sorted(analytics["ips"].items(), key=lambda x: x[1], reverse=True)[:20]

    # Get daily breakdown (last 30 days)
    daily = sorted(analytics["daily"].items())[-30:]

    # Get hourly data (last 24 hours)
    hourly = sorted(analytics["hourly"].items())[-24:]

    # Calculate time-range stats
    from datetime import timedelta
    today = date.today()
    last_7_days = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
    last_30_days = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(30)]

    requests_7_days = sum(analytics["daily"].get(d, 0) for d in last_7_days)
    requests_30_days = sum(analytics["daily"].get(d, 0) for d in last_30_days)

    # Calculate unique IPs for time ranges
    unique_ips_today = len(set(
        ip for ip, journeys in analytics.get("user_journeys", {}).items()
        for j in journeys if j.get("timestamp", "").startswith(str(today))
    )) or len(analytics["ips"])  # Fallback to total if no journey data

    # Get device/browser/OS data
    devices = sorted(analytics["devices"].items(), key=lambda x: x[1], reverse=True)
    browsers = sorted(analytics["browsers"].items(), key=lambda x: x[1], reverse=True)
    os_data = sorted(analytics["os"].items(), key=lambda x: x[1], reverse=True)
    referers = sorted(analytics["referers"].items(), key=lambda x: x[1], reverse=True)[:10]

    # Get user journeys data (for all tracked IPs)
    user_journeys_json = json.dumps(dict(analytics["user_journeys"]))

    # Prepare all data for charts
    daily_labels = json.dumps([d[0] for d in daily])
    daily_data = json.dumps([d[1] for d in daily])
    hourly_labels = json.dumps([h[0].split(" ")[1] if " " in h[0] else h[0] for h in hourly])
    hourly_data = json.dumps([h[1] for h in hourly])
    endpoint_labels = json.dumps([e[0] for e in top_endpoints])
    endpoint_data = json.dumps([e[1] for e in top_endpoints])
    ip_labels = json.dumps([ip[0][:20] + "..." if len(ip[0]) > 20 else ip[0] for ip in top_ips[:10]])
    ip_data = json.dumps([ip[1] for ip in top_ips[:10]])
    device_labels = json.dumps([d[0] for d in devices])
    device_data = json.dumps([d[1] for d in devices])
    browser_labels = json.dumps([b[0] for b in browsers])
    browser_data = json.dumps([b[1] for b in browsers])
    os_labels = json.dumps([o[0] for o in os_data])
    os_data_json = json.dumps([o[1] for o in os_data])
    referer_labels = json.dumps([r[0][:20] + "..." if len(r[0]) > 20 else r[0] for r in referers])
    referer_data = json.dumps([r[1] for r in referers])

    # All available datasets for custom builder
    all_datasets = json.dumps({
        "daily": {"labels": [d[0] for d in daily], "data": [d[1] for d in daily], "label": "Daily Requests"},
        "hourly": {"labels": [h[0].split(" ")[1] if " " in h[0] else h[0] for h in hourly], "data": [h[1] for h in hourly], "label": "Hourly Requests"},
        "endpoints": {"labels": [e[0] for e in top_endpoints], "data": [e[1] for e in top_endpoints], "label": "Endpoints"},
        "ips": {"labels": [ip[0][:20] for ip in top_ips[:10]], "data": [ip[1] for ip in top_ips[:10]], "label": "IPs"},
        "devices": {"labels": [d[0] for d in devices], "data": [d[1] for d in devices], "label": "Devices"},
        "browsers": {"labels": [b[0] for b in browsers], "data": [b[1] for b in browsers], "label": "Browsers"},
        "os": {"labels": [o[0] for o in os_data], "data": [o[1] for o in os_data], "label": "Operating Systems"},
        "referers": {"labels": [r[0][:20] for r in referers], "data": [r[1] for r in referers], "label": "Referers"}
    })

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>UFC API Analytics</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: #fff;
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{ max-width: 1600px; margin: 0 auto; }}
            h1 {{
                text-align: center;
                margin-bottom: 30px;
                font-size: 2.5rem;
                background: linear-gradient(90deg, #00d4ff, #7b2cbf);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-bottom: 25px;
            }}
            .stat-card {{
                background: rgba(255,255,255,0.1);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .stat-value {{
                font-size: 2rem;
                font-weight: bold;
                background: linear-gradient(90deg, #00d4ff, #7b2cbf);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .stat-label {{ color: #aaa; margin-top: 5px; font-size: 0.8rem; }}
            .charts-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin-bottom: 25px;
            }}
            .chart-container {{
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(255,255,255,0.1);
            }}
            .chart-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }}
            .chart-title {{ font-size: 1.1rem; color: #fff; }}
            select, button {{
                background: rgba(255,255,255,0.1);
                color: #fff;
                border: 1px solid rgba(255,255,255,0.2);
                padding: 8px 12px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 0.9rem;
            }}
            select option {{ background: #1a1a2e; color: #fff; }}
            button:hover {{ background: rgba(255,255,255,0.2); }}
            button.active {{ background: #7b2cbf; border-color: #7b2cbf; }}
            .chart-wrapper {{ position: relative; height: 250px; }}
            .custom-builder {{
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(255,255,255,0.1);
                margin-bottom: 25px;
            }}
            .builder-controls {{
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
                margin-bottom: 20px;
                align-items: center;
            }}
            .control-group {{
                display: flex;
                flex-direction: column;
                gap: 5px;
            }}
            .control-group label {{
                font-size: 0.8rem;
                color: #aaa;
            }}
            .table-container {{
                background: rgba(255,255,255,0.05);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(255,255,255,0.1);
                overflow-x: auto;
            }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); }}
            th {{ color: #00d4ff; font-weight: 600; font-size: 0.9rem; }}
            tr:hover {{ background: rgba(255,255,255,0.05); }}
            .ip-cell {{ font-family: monospace; font-size: 0.85rem; }}
            .section-title {{
                font-size: 1.3rem;
                color: #00d4ff;
                margin-bottom: 15px;
            }}
            @media (max-width: 1000px) {{
                .charts-grid {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>UFC API Analytics</h1>

            <div class="table-container" style="margin-bottom: 20px; background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(123,44,191,0.1));">
                <h3 class="section-title">📊 All Time Stats (Historical)</h3>
                <div class="stats-grid" style="margin-bottom: 0;">
                    <div class="stat-card" style="background: rgba(0,0,0,0.3);">
                        <div class="stat-value">{analytics["total_requests"]:,}</div>
                        <div class="stat-label">Total Requests</div>
                    </div>
                    <div class="stat-card" style="background: rgba(0,0,0,0.3);">
                        <div class="stat-value">{len(analytics["ips"]):,}</div>
                        <div class="stat-label">Unique IPs</div>
                    </div>
                    <div class="stat-card" style="background: rgba(0,0,0,0.3);">
                        <div class="stat-value">{len(analytics["endpoints"])}</div>
                        <div class="stat-label">Endpoints</div>
                    </div>
                    <div class="stat-card" style="background: rgba(0,0,0,0.3);">
                        <div class="stat-value">{len(analytics["devices"])}</div>
                        <div class="stat-label">Devices</div>
                    </div>
                    <div class="stat-card" style="background: rgba(0,0,0,0.3);">
                        <div class="stat-value">{len(analytics["browsers"])}</div>
                        <div class="stat-label">Browsers</div>
                    </div>
                    <div class="stat-card" style="background: rgba(0,0,0,0.3);">
                        <div class="stat-value">{len(analytics["daily"])}</div>
                        <div class="stat-label">Days Tracked</div>
                    </div>
                </div>
            </div>

            <div class="table-container" style="margin-bottom: 20px;">
                <div style="display: flex; gap: 15px; align-items: center; flex-wrap: wrap;">
                    <div class="control-group">
                        <label>📅 Date Range</label>
                        <select id="dateRangeSelect" onchange="onDateRangeChange()" style="background: #1e1e2e; color: #fff; border: 1px solid #333; padding: 10px 15px; border-radius: 5px; font-size: 1rem;">
                            <option value="today">Today</option>
                            <option value="7days">Last 7 Days</option>
                            <option value="30days">Last 30 Days</option>
                            <option value="custom">Custom Range</option>
                        </select>
                    </div>
                    <div class="control-group" id="customDateInputs" style="display: none;">
                        <label>From</label>
                        <input type="date" id="startDate" onchange="applyDateFilter()" style="background: #1e1e2e; color: #fff; border: 1px solid #333; padding: 8px 12px; border-radius: 5px;">
                    </div>
                    <div class="control-group" id="customDateInputs2" style="display: none;">
                        <label>To</label>
                        <input type="date" id="endDate" onchange="applyDateFilter()" style="background: #1e1e2e; color: #fff; border: 1px solid #333; padding: 8px 12px; border-radius: 5px;">
                    </div>
                    <div style="margin-left: auto; color: #666; font-size: 0.9rem;">
                        Showing: <span id="dateRangeLabel" style="color: #00d4ff;">Today</span>
                    </div>
                </div>
            </div>

            <h3 class="section-title" style="margin-bottom: 15px;">📈 Filtered Stats</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="statRequests">0</div>
                    <div class="stat-label">Requests</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="statUniqueIPs">0</div>
                    <div class="stat-label">Unique IPs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="statEndpoints">0</div>
                    <div class="stat-label">Endpoints Hit</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="statDevices">0</div>
                    <div class="stat-label">Device Types</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="statBrowsers">0</div>
                    <div class="stat-label">Browsers</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="statOS">0</div>
                    <div class="stat-label">OS Types</div>
                </div>
            </div>

            <div class="charts-grid">
                <div class="chart-container">
                    <div class="chart-header">
                        <span class="chart-title">Daily Requests</span>
                        <select id="dailyChartType" onchange="updateChart('daily')">
                            <option value="line">Line</option>
                            <option value="bar">Bar</option>
                        </select>
                    </div>
                    <div class="chart-wrapper"><canvas id="dailyChart"></canvas></div>
                </div>

                <div class="chart-container">
                    <div class="chart-header">
                        <span class="chart-title">Top Endpoints</span>
                        <select id="endpointsChartType" onchange="updateChart('endpoints')">
                            <option value="bar">Bar</option>
                            <option value="doughnut">Doughnut</option>
                            <option value="pie">Pie</option>
                        </select>
                    </div>
                    <div class="chart-wrapper"><canvas id="endpointsChart"></canvas></div>
                </div>

                <div class="chart-container">
                    <div class="chart-header">
                        <span class="chart-title">Requests by IP</span>
                        <select id="ipsChartType" onchange="updateChart('ips')">
                            <option value="bar">Bar</option>
                            <option value="doughnut">Doughnut</option>
                        </select>
                    </div>
                    <div class="chart-wrapper"><canvas id="ipsChart"></canvas></div>
                </div>

                <div class="chart-container">
                    <div class="chart-header">
                        <span class="chart-title">Devices</span>
                        <select id="devicesChartType" onchange="updateChart('devices')">
                            <option value="doughnut">Doughnut</option>
                            <option value="pie">Pie</option>
                            <option value="bar">Bar</option>
                        </select>
                    </div>
                    <div class="chart-wrapper"><canvas id="devicesChart"></canvas></div>
                </div>

                <div class="chart-container">
                    <div class="chart-header">
                        <span class="chart-title">Browsers</span>
                        <select id="browsersChartType" onchange="updateChart('browsers')">
                            <option value="doughnut">Doughnut</option>
                            <option value="pie">Pie</option>
                            <option value="bar">Bar</option>
                        </select>
                    </div>
                    <div class="chart-wrapper"><canvas id="browsersChart"></canvas></div>
                </div>

                <div class="chart-container">
                    <div class="chart-header">
                        <span class="chart-title">Operating Systems</span>
                        <select id="osChartType" onchange="updateChart('os')">
                            <option value="doughnut">Doughnut</option>
                            <option value="pie">Pie</option>
                            <option value="bar">Bar</option>
                        </select>
                    </div>
                    <div class="chart-wrapper"><canvas id="osChart"></canvas></div>
                </div>
            </div>

            <!-- Custom Visualization Builder -->
            <div class="custom-builder">
                <h3 class="section-title">Custom Visualization Builder</h3>
                <div class="builder-controls">
                    <div class="control-group">
                        <label>Data Source</label>
                        <select id="customData" onchange="updateCustomChart()">
                            <option value="daily">Daily Requests</option>
                            <option value="hourly">Hourly Requests</option>
                            <option value="endpoints">Endpoints</option>
                            <option value="ips">IP Addresses</option>
                            <option value="devices">Devices</option>
                            <option value="browsers">Browsers</option>
                            <option value="os">Operating Systems</option>
                            <option value="referers">Referers</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Chart Type</label>
                        <select id="customChartType" onchange="updateCustomChart()">
                            <option value="bar">Bar Chart</option>
                            <option value="line">Line Chart</option>
                            <option value="pie">Pie Chart</option>
                            <option value="doughnut">Doughnut Chart</option>
                            <option value="polarArea">Polar Area</option>
                            <option value="radar">Radar Chart</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Orientation</label>
                        <select id="customOrientation" onchange="updateCustomChart()">
                            <option value="vertical">Vertical</option>
                            <option value="horizontal">Horizontal</option>
                        </select>
                    </div>
                    <div class="control-group">
                        <label>Show Legend</label>
                        <select id="customLegend" onchange="updateCustomChart()">
                            <option value="true">Yes</option>
                            <option value="false">No</option>
                        </select>
                    </div>
                </div>
                <div class="chart-wrapper" style="height: 350px;"><canvas id="customChart"></canvas></div>
            </div>

            <div class="table-container">
                <h3 class="section-title">All IPs & Request Counts</h3>
                <table>
                    <thead>
                        <tr><th>#</th><th>IP Address</th><th>Requests</th><th>% of Total</th></tr>
                    </thead>
                    <tbody>
                        {"".join(f'<tr><td>{i+1}</td><td class="ip-cell">{ip}</td><td>{count:,}</td><td>{(count/max(analytics["total_requests"],1)*100):.1f}%</td></tr>' for i, (ip, count) in enumerate(top_ips))}
                    </tbody>
                </table>
            </div>

            <div class="table-container" style="grid-column: 1 / -1;">
                <h3 class="section-title">🔍 User Journeys (by IP)</h3>
                <div style="margin-bottom: 15px;">
                    <label style="color: #aaa;">Select IP: </label>
                    <select id="journeyIpSelect" onchange="showJourney()" style="background: #1e1e2e; color: #fff; border: 1px solid #333; padding: 8px 12px; border-radius: 5px; min-width: 200px;">
                        <option value="">-- Select an IP --</option>
                        {"".join(f'<option value="{ip}">{ip} ({count} requests)</option>' for ip, count in top_ips)}
                    </select>
                </div>
                <div id="journeyContainer" style="max-height: 400px; overflow-y: auto;">
                    <p style="color: #666;">Select an IP address to view their journey</p>
                </div>
            </div>
        </div>

        <script>
            const colors = ['#00d4ff', '#7b2cbf', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96c93d', '#f9ca24', '#f0932b', '#eb4d4b', '#6c5ce7'];
            const allDatasets = {all_datasets};
            const userJourneys = {user_journeys_json};
            const dailyData = {json.dumps(dict(analytics["daily"]))};

            // Flatten all journeys into a single array with IP included
            let allRequests = [];
            for (const [ip, journeys] of Object.entries(userJourneys)) {{
                for (const j of journeys) {{
                    allRequests.push({{ ...j, ip: ip }});
                }}
            }}

            // Current filtered data
            let filteredRequests = [];

            // Get date string for N days ago
            function getDateNDaysAgo(n) {{
                const d = new Date();
                d.setDate(d.getDate() - n);
                return d.toISOString().split('T')[0];
            }}

            // Filter requests by date range
            function filterByDateRange(startDate, endDate) {{
                return allRequests.filter(r => {{
                    const date = r.date || r.timestamp?.split(' ')[0];
                    return date >= startDate && date <= endDate;
                }});
            }}

            // Calculate stats from filtered requests
            function calculateStats(requests) {{
                const uniqueIPs = new Set(requests.map(r => r.ip));
                const uniqueEndpoints = new Set(requests.map(r => r.endpoint));
                const uniqueDevices = new Set(requests.map(r => r.device).filter(Boolean));
                const uniqueBrowsers = new Set(requests.map(r => r.browser).filter(Boolean));
                const uniqueOS = new Set(requests.map(r => r.os).filter(Boolean));

                return {{
                    requests: requests.length,
                    uniqueIPs: uniqueIPs.size,
                    endpoints: uniqueEndpoints.size,
                    devices: uniqueDevices.size,
                    browsers: uniqueBrowsers.size,
                    os: uniqueOS.size
                }};
            }}

            // Update stat cards
            function updateStatCards(stats) {{
                document.getElementById('statRequests').textContent = stats.requests.toLocaleString();
                document.getElementById('statUniqueIPs').textContent = stats.uniqueIPs.toLocaleString();
                document.getElementById('statEndpoints').textContent = stats.endpoints.toLocaleString();
                document.getElementById('statDevices').textContent = stats.devices.toLocaleString();
                document.getElementById('statBrowsers').textContent = stats.browsers.toLocaleString();
                document.getElementById('statOS').textContent = stats.os.toLocaleString();
            }}

            // Get aggregated data for charts
            function getChartData(requests) {{
                const endpoints = {{}};
                const ips = {{}};
                const devices = {{}};
                const browsers = {{}};
                const osData = {{}};
                const daily = {{}};

                requests.forEach(r => {{
                    endpoints[r.endpoint] = (endpoints[r.endpoint] || 0) + 1;
                    ips[r.ip] = (ips[r.ip] || 0) + 1;
                    if (r.device) devices[r.device] = (devices[r.device] || 0) + 1;
                    if (r.browser) browsers[r.browser] = (browsers[r.browser] || 0) + 1;
                    if (r.os) osData[r.os] = (osData[r.os] || 0) + 1;
                    const date = r.date || r.timestamp?.split(' ')[0];
                    if (date) daily[date] = (daily[date] || 0) + 1;
                }});

                return {{ endpoints, ips, devices, browsers, osData, daily }};
            }}

            // Update all charts with filtered data
            function updateAllCharts(requests) {{
                const data = getChartData(requests);

                // Sort and limit data
                const sortedEndpoints = Object.entries(data.endpoints).sort((a, b) => b[1] - a[1]).slice(0, 10);
                const sortedIPs = Object.entries(data.ips).sort((a, b) => b[1] - a[1]).slice(0, 10);
                const sortedDaily = Object.entries(data.daily).sort((a, b) => a[0].localeCompare(b[0]));

                // Update chart data
                chartData.endpoints = {{ labels: sortedEndpoints.map(e => e[0]), data: sortedEndpoints.map(e => e[1]) }};
                chartData.ips = {{ labels: sortedIPs.map(e => e[0].substring(0, 20)), data: sortedIPs.map(e => e[1]) }};
                chartData.devices = {{ labels: Object.keys(data.devices), data: Object.values(data.devices) }};
                chartData.browsers = {{ labels: Object.keys(data.browsers), data: Object.values(data.browsers) }};
                chartData.os = {{ labels: Object.keys(data.osData), data: Object.values(data.osData) }};
                chartData.daily = {{ labels: sortedDaily.map(e => e[0]), data: sortedDaily.map(e => e[1]) }};

                // Recreate charts
                createChart('daily', document.getElementById('dailyChartType')?.value || 'line', chartData.daily.labels, chartData.daily.data);
                createChart('endpoints', document.getElementById('endpointsChartType')?.value || 'bar', chartData.endpoints.labels, chartData.endpoints.data, true);
                createChart('ips', document.getElementById('ipsChartType')?.value || 'bar', chartData.ips.labels, chartData.ips.data);
                createChart('devices', document.getElementById('devicesChartType')?.value || 'doughnut', chartData.devices.labels, chartData.devices.data);
                createChart('browsers', document.getElementById('browsersChartType')?.value || 'doughnut', chartData.browsers.labels, chartData.browsers.data);
                createChart('os', document.getElementById('osChartType')?.value || 'doughnut', chartData.os.labels, chartData.os.data);
            }}

            // Handle date range dropdown change
            function onDateRangeChange() {{
                const select = document.getElementById('dateRangeSelect');
                const customInputs = document.getElementById('customDateInputs');
                const customInputs2 = document.getElementById('customDateInputs2');
                const label = document.getElementById('dateRangeLabel');

                if (select.value === 'custom') {{
                    customInputs.style.display = 'flex';
                    customInputs2.style.display = 'flex';
                    label.textContent = 'Custom Range';
                }} else {{
                    customInputs.style.display = 'none';
                    customInputs2.style.display = 'none';
                    applyDateFilter();
                }}
            }}

            // Apply the date filter
            function applyDateFilter() {{
                const select = document.getElementById('dateRangeSelect');
                const today = new Date().toISOString().split('T')[0];
                let startDate, endDate, labelText;

                switch (select.value) {{
                    case 'today':
                        startDate = endDate = today;
                        labelText = 'Today';
                        break;
                    case '7days':
                        startDate = getDateNDaysAgo(6);
                        endDate = today;
                        labelText = 'Last 7 Days';
                        break;
                    case '30days':
                        startDate = getDateNDaysAgo(29);
                        endDate = today;
                        labelText = 'Last 30 Days';
                        break;
                    case 'custom':
                        startDate = document.getElementById('startDate').value;
                        endDate = document.getElementById('endDate').value;
                        if (!startDate || !endDate) return;
                        labelText = `${{startDate}} to ${{endDate}}`;
                        break;
                }}

                document.getElementById('dateRangeLabel').textContent = labelText;

                // Filter and update
                filteredRequests = filterByDateRange(startDate, endDate);
                const stats = calculateStats(filteredRequests);
                updateStatCards(stats);
                updateAllCharts(filteredRequests);
            }}

            // Initialize dates
            document.addEventListener('DOMContentLoaded', function() {{
                const today = new Date().toISOString().split('T')[0];
                document.getElementById('endDate').value = today;
                document.getElementById('startDate').value = today;
                // Apply initial filter (today)
                setTimeout(applyDateFilter, 100);
            }});

            function showJourney() {{
                const ip = document.getElementById('journeyIpSelect').value;
                const container = document.getElementById('journeyContainer');

                if (!ip || !userJourneys[ip] || userJourneys[ip].length === 0) {{
                    container.innerHTML = '<p style="color: #666;">No journey data available for this IP</p>';
                    return;
                }}

                const journey = userJourneys[ip];
                let html = '<table style="width: 100%;"><thead><tr><th>Time</th><th>Method</th><th>Endpoint</th><th>Referer</th></tr></thead><tbody>';

                journey.slice().reverse().forEach((entry, i) => {{
                    const rowColor = i % 2 === 0 ? 'rgba(255,255,255,0.02)' : 'transparent';
                    html += `<tr style="background: ${{rowColor}}">
                        <td style="white-space: nowrap; color: #888;">${{entry.timestamp}}</td>
                        <td><span style="background: #7b2cbf; padding: 2px 8px; border-radius: 3px; font-size: 11px;">${{entry.method}}</span></td>
                        <td style="color: #00d4ff;">${{entry.endpoint}}</td>
                        <td style="color: #666; font-size: 12px;">${{entry.referer || 'Direct'}}</td>
                    </tr>`;
                }});

                html += '</tbody></table>';
                container.innerHTML = html;
            }}

            const chartData = {{
                daily: {{ labels: {daily_labels}, data: {daily_data} }},
                endpoints: {{ labels: {endpoint_labels}, data: {endpoint_data} }},
                ips: {{ labels: {ip_labels}, data: {ip_data} }},
                devices: {{ labels: {device_labels}, data: {device_data} }},
                browsers: {{ labels: {browser_labels}, data: {browser_data} }},
                os: {{ labels: {os_labels}, data: {os_data_json} }}
            }};

            const charts = {{}};

            function createChart(id, type, labels, data, horizontal = false) {{
                const ctx = document.getElementById(id + 'Chart');
                if (charts[id]) charts[id].destroy();

                const isCircular = ['pie', 'doughnut', 'polarArea', 'radar'].includes(type);

                charts[id] = new Chart(ctx, {{
                    type: type,
                    data: {{
                        labels: labels,
                        datasets: [{{
                            label: 'Count',
                            data: data,
                            backgroundColor: isCircular ? colors : colors[0],
                            borderColor: type === 'line' ? colors[0] : undefined,
                            fill: type === 'line',
                            tension: 0.4
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        indexAxis: horizontal && type === 'bar' ? 'y' : 'x',
                        plugins: {{
                            legend: {{ display: isCircular, position: 'right', labels: {{ color: '#aaa', font: {{ size: 11 }} }} }}
                        }},
                        scales: isCircular ? {{}} : {{
                            y: {{ beginAtZero: true, grid: {{ color: 'rgba(255,255,255,0.1)' }}, ticks: {{ color: '#aaa' }} }},
                            x: {{ grid: {{ display: false }}, ticks: {{ color: '#aaa', maxRotation: 45 }} }}
                        }}
                    }}
                }});
            }}

            function updateChart(name) {{
                const type = document.getElementById(name + 'ChartType').value;
                const d = chartData[name];
                const horizontal = name === 'endpoints';
                createChart(name, type, d.labels, d.data, horizontal);
            }}

            // Initialize all charts
            createChart('daily', 'line', chartData.daily.labels, chartData.daily.data);
            createChart('endpoints', 'bar', chartData.endpoints.labels, chartData.endpoints.data, true);
            createChart('ips', 'bar', chartData.ips.labels, chartData.ips.data);
            createChart('devices', 'doughnut', chartData.devices.labels, chartData.devices.data);
            createChart('browsers', 'doughnut', chartData.browsers.labels, chartData.browsers.data);
            createChart('os', 'doughnut', chartData.os.labels, chartData.os.data);

            // Custom Chart Builder
            let customChartInstance = null;

            function updateCustomChart() {{
                const dataSource = document.getElementById('customData').value;
                const chartType = document.getElementById('customChartType').value;
                const orientation = document.getElementById('customOrientation').value;
                const showLegend = document.getElementById('customLegend').value === 'true';

                const dataset = allDatasets[dataSource];
                const ctx = document.getElementById('customChart');

                if (customChartInstance) customChartInstance.destroy();

                const isCircular = ['pie', 'doughnut', 'polarArea', 'radar'].includes(chartType);
                const isHorizontal = orientation === 'horizontal' && chartType === 'bar';

                customChartInstance = new Chart(ctx, {{
                    type: chartType,
                    data: {{
                        labels: dataset.labels,
                        datasets: [{{
                            label: dataset.label,
                            data: dataset.data,
                            backgroundColor: isCircular ? colors : colors.slice(0, dataset.data.length),
                            borderColor: chartType === 'line' ? colors[0] : chartType === 'radar' ? colors[0] : undefined,
                            fill: chartType === 'line' || chartType === 'radar',
                            tension: 0.4
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        indexAxis: isHorizontal ? 'y' : 'x',
                        plugins: {{
                            legend: {{ display: showLegend, position: 'top', labels: {{ color: '#aaa' }} }},
                            title: {{ display: true, text: dataset.label, color: '#fff', font: {{ size: 14 }} }}
                        }},
                        scales: isCircular ? {{}} : {{
                            y: {{ beginAtZero: true, grid: {{ color: 'rgba(255,255,255,0.1)' }}, ticks: {{ color: '#aaa' }} }},
                            x: {{ grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#aaa', maxRotation: 45 }} }}
                        }}
                    }}
                }});
            }}

            // Initialize custom chart
            updateCustomChart();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/analytics/json", tags=["System"], include_in_schema=False)
async def get_analytics_json(request: Request, secret: str = Query(..., description="Analytics secret key")):
    """Get raw analytics data as JSON."""
    if secret != ANALYTICS_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret key")

    save_analytics()
    top_endpoints = sorted(analytics["endpoints"].items(), key=lambda x: x[1], reverse=True)[:10]
    top_ips = sorted(analytics["ips"].items(), key=lambda x: x[1], reverse=True)[:20]
    daily = sorted(analytics["daily"].items(), reverse=True)[:30]
    hourly = sorted(analytics["hourly"].items(), reverse=True)[:24]
    devices = sorted(analytics["devices"].items(), key=lambda x: x[1], reverse=True)
    browsers = sorted(analytics["browsers"].items(), key=lambda x: x[1], reverse=True)
    os_data = sorted(analytics["os"].items(), key=lambda x: x[1], reverse=True)
    referers = sorted(analytics["referers"].items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "summary": {
            "total_requests": analytics["total_requests"],
            "requests_today": analytics["requests_today"],
            "unique_ips": len(analytics["ips"]),
            "unique_devices": len(analytics["devices"]),
            "unique_browsers": len(analytics["browsers"]),
            "today": analytics["today"]
        },
        "top_endpoints": [{"endpoint": e, "count": c} for e, c in top_endpoints],
        "top_users_by_ip": [{"ip": ip, "requests": c} for ip, c in top_ips],
        "daily_breakdown": [{"date": d, "requests": c} for d, c in daily],
        "hourly_breakdown": [{"hour": h, "requests": c} for h, c in hourly],
        "devices": [{"device": d, "count": c} for d, c in devices],
        "browsers": [{"browser": b, "count": c} for b, c in browsers],
        "operating_systems": [{"os": o, "count": c} for o, c in os_data],
        "referers": [{"referer": r, "count": c} for r, c in referers]
    }


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health", tags=["System"])
@limiter.limit(RATE_LIMIT)
async def health_check(request: Request):
    return {
        "status": "healthy",
        "fighters_loaded": len(FIGHTERS),
        "events_loaded": len(EVENTS),
        "fights_loaded": len(FIGHTS),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/", tags=["System"])
@limiter.limit(RATE_LIMIT)
async def root(request: Request):
    return {
        "message": "UFC Stats API",
        "version": "1.0.0",
        "docs": "/docs",
        "rate_limit": {
            "limit": "100 requests/day per IP",
            "burst": "100 requests/minute",
            "need_more": "Visit https://aristotle.me"
        },
        "endpoints": {
            "fighters": "/api/fighters",
            "fighter_detail": "/api/fighters/{id}",
            "fighter_stats": "/api/fighters/{id}/stats",
            "compare": "/api/fighters/compare?fighter1=ID&fighter2=ID",
            "events": "/api/events",
            "event_detail": "/api/events/{id}",
            "fights": "/api/fights",
            "fight_detail": "/api/fights/{id}",
            "stat_leaders": "/api/stats/leaders",
            "overview": "/api/stats/overview"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
