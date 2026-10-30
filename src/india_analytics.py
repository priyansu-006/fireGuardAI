"""India Forest Fire Incident Analytics and Vulnerability Modeling Module.

Processes historical Indian State & UT wildfire incident data (2008-2011),
computes 3-year averages, peak annual incidents, vulnerability scores,
and regional forest belt coordinates for all 35 States/UTs.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import BASE_DIR, RAW_DATA_DIR, PLOTS_DIR

INDIA_DATA_PATH = RAW_DATA_DIR / "india_forest_fires_by_state.csv"
INDIA_PLOT_PATH = PLOTS_DIR / "india_state_fire_incident_rankings.png"

# Accurate geographic coordinate mappings and primary forest belts for Indian States
STATE_METADATA = {
    "Mizoram": {
        "belt": "Aizawl & Dampa Tropical Bamboo Ridge",
        "sector": "Sector East (Mizo Hills Dense Bamboo Belt)",
        "lat": 23.72, "lon": 92.71,
    },
    "Madhya Pradesh": {
        "belt": "Satpura-Vindhya Sal & Teak Forest Reserve",
        "sector": "Sector Central (Kanha-Bandhavgarh Corridor)",
        "lat": 22.71, "lon": 78.43,
    },
    "Chhatisgarh": {
        "belt": "Bastar & Kanger Valley Dense Sal Reserve",
        "sector": "Sector Central-East (Dandakaranya Deciduous Basin)",
        "lat": 19.07, "lon": 82.02,
    },
    "Assam": {
        "belt": "Karbi Anglong & Kaziranga Foothill Woodlands",
        "sector": "Sector North-East (Brahmaputra Valley Wet Evergreen)",
        "lat": 26.20, "lon": 92.93,
    },
    "Andhra Pradesh": {
        "belt": "Seshachalam & Nallamala Forest Hills",
        "sector": "Sector South-East (Eastern Ghats Red Sanders Belt)",
        "lat": 13.68, "lon": 79.35,
    },
    "Maharashtra": {
        "belt": "Chandrapur & Melghat Dry Deciduous Zone",
        "sector": "Sector West (Tadoba-Andhari Wilderness Corridor)",
        "lat": 20.95, "lon": 77.75,
    },
    "Manipur": {
        "belt": "Senapati & Churachandpur Pine-Oak Forest",
        "sector": "Sector East (Manipur Valley Highland Forest)",
        "lat": 24.81, "lon": 93.93,
    },
    "Meghalaya": {
        "belt": "Garo & Khasi Sacred Grove Rainforest",
        "sector": "Sector North-East (Shillong Plateau Pine Corridor)",
        "lat": 25.57, "lon": 91.89,
    },
    "Nagaland": {
        "belt": "Dzukou Valley & Saramati Subtropical Forest",
        "sector": "Sector East (Naga Hills Montane Forest)",
        "lat": 25.67, "lon": 94.10,
    },
    "Orissa": {
        "belt": "Similipal Biosphere & Mayurbhanj Sal Forest",
        "sector": "Sector East (North-Eastern Ghats Moist Deciduous)",
        "lat": 21.85, "lon": 86.34,
    },
    "Jharkhand": {
        "belt": "Palamu & Chota Nagpur Plateau Forests",
        "sector": "Sector Central-East (Dalma-Saranda Sal Forest)",
        "lat": 23.61, "lon": 85.27,
    },
    "Uttarakhand": {
        "belt": "Western Himalayan Chir Pine & Deodar Belt",
        "sector": "Sector North (Nainital-Almora-Pauri Forest Ridge)",
        "lat": 29.60, "lon": 79.66,
    },
    "Arunachal Pradesh": {
        "belt": "Namdapha & Eastern Himalayan Broadleaf Forest",
        "sector": "Sector North-East (Sub-Alpine Conifer Belt)",
        "lat": 27.08, "lon": 93.60,
    },
    "Karnataka": {
        "belt": "Bandipur & Nagarhole Western Ghats Reserve",
        "sector": "Sector South (Nilgiri Biosphere Deciduous Belt)",
        "lat": 11.66, "lon": 76.63,
    },
    "Himachal Pradesh": {
        "belt": "Kangra & Solan Pine-Conifer Ridge",
        "sector": "Sector North-West (Shimla-Kullu Alpine Corridor)",
        "lat": 31.10, "lon": 77.17,
    },
    "Uttar Pradesh": {
        "belt": "Dudhwa Terai & Shivalik Foothill Forest",
        "sector": "Sector North (Lakhimpur-Kheri Sal Swamp Forest)",
        "lat": 28.32, "lon": 80.60,
    },
    "West Bengal": {
        "belt": "Dooars & Sunderbans Mangrove/Sal Forest",
        "sector": "Sector East (Buxa-Gorumara Foothills)",
        "lat": 26.53, "lon": 88.71,
    },
    "Gujarat": {
        "belt": "Gir Dry Deciduous & Dangs Teak Forest",
        "sector": "Sector West (Saurashtra Scrubland Belt)",
        "lat": 21.12, "lon": 70.82,
    },
    "Kerala": {
        "belt": "Silent Valley & Wayanad Evergreen Belt",
        "sector": "Sector South (Southern Western Ghats Rainforest)",
        "lat": 11.05, "lon": 76.43,
    },
    "Tamil nadu": {
        "belt": "Mudumalai & Anamalai Tiger Reserves",
        "sector": "Sector South (Southern Eastern Ghats Scrub Forest)",
        "lat": 11.56, "lon": 76.53,
    },
    "Tripura": {
        "belt": "Sepahijala & Gumti Semievergreen Forest",
        "sector": "Sector East (Tripura Lowland Hills)",
        "lat": 23.83, "lon": 91.28,
    },
    "Rajasthan": {
        "belt": "Ranthambore & Aravalli Dry Deciduous Forest",
        "sector": "Sector West (Dhonk & Khair Dry Woodland)",
        "lat": 26.01, "lon": 76.50,
    },
    "Bihar": {
        "belt": "Valmiki Tiger Reserve Terai Belt",
        "sector": "Sector North (Champaran Mixed Deciduous)",
        "lat": 27.42, "lon": 84.15,
    },
    "Goa": {
        "belt": "Bhagwan Mahavir & Mollem Western Ghats",
        "sector": "Sector South-West (Semi-Evergreen Forest)",
        "lat": 15.33, "lon": 74.22,
    },
    "Jammu and Kashmir": {
        "belt": "Pir Panjal & Dachigam Coniferous Forest",
        "sector": "Sector North (Kashmir Temperate Pine Corridor)",
        "lat": 34.13, "lon": 74.90,
    },
    "Punjab": {
        "belt": "Shivalik Hills Dry Deciduous Pockets",
        "sector": "Sector North (Hoshiarpur-Ropar Foot Forest)",
        "lat": 31.52, "lon": 75.91,
    },
    "Haryana": {
        "belt": "Kalesar National Park Sal Forest",
        "sector": "Sector North (Yamunanagar Shivalik Belt)",
        "lat": 30.33, "lon": 77.58,
    },
    "Sikkim": {
        "belt": "Khangchendzonga Sub-Alpine Rhododendron",
        "sector": "Sector North-East (Eastern Himalayan High Alpine)",
        "lat": 27.53, "lon": 88.51,
    },
    "Andaman and Nicobar": {
        "belt": "Great Nicobar & Saddle Peak Tropical Rainforest",
        "sector": "Sector Island (Bay of Bengal Evergreen Canopy)",
        "lat": 11.74, "lon": 92.65,
    },
    "Delhi": {
        "belt": "Asola Bhatti Wildlife Sanctuary & Southern Ridge",
        "sector": "Sector NCR (Northern Aravalli Scrubland Corridor)",
        "lat": 28.51, "lon": 77.25,
    },
    "Chandigarh": {
        "belt": "Sukhna Wildlife Sanctuary & Shivalik Foothills",
        "sector": "Sector North (Sub-Himalayan Green Reserve)",
        "lat": 30.75, "lon": 76.82,
    },
    "Dadra and Nagar haveli": {
        "belt": "Satmaliya Deer Park & Daman Ganga Riverine Forest",
        "sector": "Sector West (Western Ghats Border Woodland)",
        "lat": 20.27, "lon": 73.01,
    },
    "Daman and Diu": {
        "belt": "Fudam Bird Sanctuary & Coastal Scrub Forest",
        "sector": "Sector Coastal (Arabian Sea Littoral Zone)",
        "lat": 20.71, "lon": 70.92,
    },
    "Lakshadweep": {
        "belt": "Pitti Island Coral Atoll Marine & Terrestrial Preserve",
        "sector": "Sector Island (Arabian Sea Coral Canopy)",
        "lat": 10.57, "lon": 72.63,
    },
    "Puducherry": {
        "belt": "Ousteri Lake Forest Sanctuary & Coromandel Coast",
        "sector": "Sector South-East (Coastal Littoral Scrub)",
        "lat": 11.95, "lon": 79.74,
    },
}

# State name aliases for flexible frontend inputs
STATE_ALIASES = {
    "odisha": "Orissa",
    "chhattisgarh": "Chhatisgarh",
    "tamil nadu": "Tamil nadu",
    "dadra & nagar haveli": "Dadra and Nagar haveli",
    "dadra and nagar haveli": "Dadra and Nagar haveli",
    "daman & diu": "Daman and Diu",
    "daman and diu": "Daman and Diu",
    "jammu & kashmir": "Jammu and Kashmir",
    "jammu and kashmir": "Jammu and Kashmir",
    "andaman & nicobar": "Andaman and Nicobar",
    "andaman and nicobar": "Andaman and Nicobar",
    "andaman and nicobar islands": "Andaman and Nicobar",
}


def load_india_fire_dataset() -> pd.DataFrame:
    """Load and process India state-wise annual forest fire incidents."""
    if not INDIA_DATA_PATH.exists():
        raise FileNotFoundError(f"India forest fire dataset not found at {INDIA_DATA_PATH}")

    df = pd.read_csv(INDIA_DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    # Convert numeric columns
    year_cols = ["2010-2011", "2009-10", "2008-09"]
    for y in year_cols:
        df[y] = pd.to_numeric(df[y], errors="coerce").fillna(0).astype(int)

    # Calculate Key Analytical Metrics
    df["3Year_Average"] = df[year_cols].mean(axis=1).round(1)
    df["Peak_Annual_Fires"] = df[year_cols].max(axis=1)
    df["Total_3Year_Fires"] = df[year_cols].sum(axis=1)
    df["Trend_Change"] = df["2010-2011"] - df["2008-09"]

    # Vulnerability Index (0 - 100 normalized against peak state)
    max_peak = df["Peak_Annual_Fires"].max()
    df["Vulnerability_Score"] = ((df["Peak_Annual_Fires"] / max_peak) * 100).round(1)

    # Classify Risk Tier
    def assign_tier(row):
        avg = row["3Year_Average"]
        if avg >= 2000:
            return "Extreme Fire-Prone Zone (Tier 4)"
        elif avg >= 500:
            return "High Fire-Prone Zone (Tier 3)"
        elif avg >= 100:
            return "Moderate Fire-Prone Zone (Tier 2)"
        else:
            return "Low / Negligible Fire Zone (Tier 1)"

    df["Risk_Tier"] = df.apply(assign_tier, axis=1)

    return df


def plot_india_state_rankings(df: Optional[pd.DataFrame] = None) -> Path:
    """Generate and save publication-grade ranking plot of Indian states by wildfire incidents."""
    if df is None:
        df = load_india_fire_dataset()

    # Filter out union territories with 0 incidents for clarity
    active_df = df[df["3Year_Average"] > 10].sort_values(by="3Year_Average", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    
    # Color palette based on vulnerability
    colors = [
        "#c0392b" if tier.startswith("Extreme")
        else "#e67e22" if tier.startswith("High")
        else "#f39c12" if tier.startswith("Moderate")
        else "#27ae60"
        for tier in active_df["Risk_Tier"]
    ]

    bars = ax.barh(active_df["States/UTs"], active_df["3Year_Average"], color=colors, edgecolor="black", alpha=0.85)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 50,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width):,}",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_title("Annual Average Forest Fire Incidents Across Indian States (2008–2011)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Average Annual Fire Incidents", fontweight="bold")
    ax.set_ylabel("State / Territory", fontweight="bold")
    ax.set_xlim(0, active_df["3Year_Average"].max() * 1.15)
    plt.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(INDIA_PLOT_PATH, dpi=300)
    plt.close()
    print(f"[IndiaAnalytics] Saved state ranking plot to {INDIA_PLOT_PATH}")
    return INDIA_PLOT_PATH


def get_state_profile(state_name: str) -> Dict[str, Any]:
    """Retrieve detailed historical wildfire profile and forest belt for an Indian State."""
    df = load_india_fire_dataset()
    query = state_name.strip().lower()
    # Normalize aliases
    normalized_name = STATE_ALIASES.get(query, query)
    
    match = df[df["States/UTs"].str.lower() == normalized_name]
    if match.empty:
        # Try substring match
        match = df[df["States/UTs"].str.lower().str.contains(query, regex=False)]

    if match.empty:
        # Default fallback
        return {
            "is_indian_state": False,
            "state_name": state_name,
            "vulnerability_score": 50.0,
            "historical_annual_avg": 500,
            "peak_annual_fires": 1000,
            "risk_tier": "Standard Regional Baseline",
            "primary_forest_belt": f"{state_name} Forest Reserve",
            "sector": "Sector Central",
            "coordinates": {"lat": 22.0, "lon": 79.0},
        }

    row = match.iloc[0]
    s_name = row["States/UTs"]
    meta = STATE_METADATA.get(s_name, {
        "belt": f"{s_name} State Forest Reserve",
        "sector": f"Sector Central ({s_name} Woodland Corridor)",
        "lat": 22.0, "lon": 79.0,
    })

    return {
        "is_indian_state": True,
        "state_name": s_name,
        "f_2010_2011": int(row["2010-2011"]),
        "f_2009_2010": int(row["2009-10"]),
        "f_2008_2009": int(row["2008-09"]),
        "historical_annual_avg": float(row["3Year_Average"]),
        "peak_annual_fires": int(row["Peak_Annual_Fires"]),
        "total_3year_fires": int(row["Total_3Year_Fires"]),
        "trend_change": int(row["Trend_Change"]),
        "vulnerability_score": float(row["Vulnerability_Score"]),
        "risk_tier": str(row["Risk_Tier"]),
        "primary_forest_belt": meta["belt"],
        "sector": meta["sector"],
        "coordinates": {"lat": meta["lat"], "lon": meta["lon"]},
    }


def get_all_states_ranking() -> List[Dict[str, Any]]:
    """Return all 35 Indian states ranked by historical wildfire vulnerability."""
    df = load_india_fire_dataset()
    sorted_df = df.sort_values(by="3Year_Average", ascending=False).reset_index(drop=True)
    # Add rank column 1 to 35
    sorted_df["Rank"] = range(1, len(sorted_df) + 1)
    return sorted_df.to_dict(orient="records")


def get_national_summary_stats() -> Dict[str, Any]:
    """Compute aggregate national wildfire statistics across all 35 States/UTs."""
    df = load_india_fire_dataset()
    total_3yr = int(df["Total_3Year_Fires"].sum())
    avg_per_yr = round(float(total_3yr / 3.0), 1)
    top_state = df.sort_values(by="Peak_Annual_Fires", ascending=False).iloc[0]
    second_state = df.sort_values(by="Peak_Annual_Fires", ascending=False).iloc[1]
    
    tier_counts = df["Risk_Tier"].value_counts().to_dict()

    return {
        "total_states_monitored": len(df),
        "national_total_3yr_incidents": total_3yr,
        "national_annual_average_incidents": avg_per_yr,
        "highest_fire_prone_state": {
            "name": top_state["States/UTs"],
            "peak_fires": int(top_state["Peak_Annual_Fires"]),
            "three_year_avg": float(top_state["3Year_Average"]),
        },
        "second_highest_state": {
            "name": second_state["States/UTs"],
            "peak_fires": int(second_state["Peak_Annual_Fires"]),
            "three_year_avg": float(second_state["3Year_Average"]),
        },
        "tier_distribution": tier_counts,
    }


if __name__ == "__main__":
    df = load_india_fire_dataset()
    print("India Dataset Processed successfully. Top 5 fire-prone states:")
    print(df.sort_values(by="3Year_Average", ascending=False)[["States/UTs", "3Year_Average", "Peak_Annual_Fires", "Risk_Tier"]].head(8).to_string())
    plot_india_state_rankings(df)
