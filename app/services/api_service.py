import httpx
from app.config.settings import settings

async def fetch_college_scorecard(query: str):
    """
    Asynchronous fetch for US university data.
    """
    if not settings.COLLEGE_API_KEY:
        return "College API Key missing."

    url = "https://api.data.gov/ed/collegescorecard/v1/schools.json"
    params = {
        "api_key": settings.COLLEGE_API_KEY,
        "school.name": query,
        "fields": "school.name,latest.cost.tuition.out_of_state,latest.admissions.admission_rate.overall"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params, timeout=15.0)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if not results:
                    return f"No official records found for {query}."
                
                output = []
                for school in results:
                    name = school.get("school.name")
                    tuition = school.get("latest.cost.tuition.out_of_state", "N/A")
                    admission = school.get("latest.admissions.admission_rate.overall", "N/A")
                    output.append(f"University: {name} | Tuition: ${tuition} | Admission Rate: {admission}")
                
                return "\n".join(output)
            return f"API Error: {response.status_code}"
        except Exception as e:
            return f"External API request failed: {str(e)}"
