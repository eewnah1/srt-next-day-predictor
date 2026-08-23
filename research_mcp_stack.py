"""Research MCP adapters."""
from datetime import datetime, timezone
def fetch_week_ahead_us_apac():
    try:
        import requests
        probes=[]
        for name,url in [("te","https://mcp.tradingeconomics.com"),("finnhub","https://mcp.finnhub.io/mcp"),("earningscalls","https://mcp.earningscalls.dev/mcp")]:
            try:
                r=requests.get(url,timeout=3); probes.append({"name":name,"http_status":r.status_code,"reachable":r.status_code<500})
            except Exception as e:
                probes.append({"name":name,"reachable":False,"error":str(e)[:120]})
        return {"name":"week_ahead_us_apac","as_of":datetime.now(timezone.utc).isoformat(),"tz_display":"SGT","mcp_probes":probes}
    except Exception as exc:
        return {"status":"error","error":str(exc)[:200]}
