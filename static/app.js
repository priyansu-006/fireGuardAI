/**
 * ForestFire ML: Frontend Interactive Controller
 */

function initDashboard() {
  const form = document.getElementById("prediction-form");
  const resultPlaceholder = document.getElementById("result-placeholder");
  const resultContent = document.getElementById("result-content");
  const riskBadge = document.getElementById("risk-badge");
  const badgeIcon = document.getElementById("badge-icon");
  const riskTitle = document.getElementById("risk-title");
  const riskDesc = document.getElementById("risk-desc");
  const confidencePct = document.getElementById("confidence-pct");
  const progressBarFill = document.getElementById("progress-bar-fill");
  const probFire = document.getElementById("prob-fire");
  const probNotFire = document.getElementById("prob-not-fire");
  const factorsList = document.getElementById("factors-list");
  const predictBtn = document.getElementById("predict-btn");

  const navBtns = document.querySelectorAll(".nav-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  // Synchronize range sliders
  const sliderConfig = [
    { id: "input-temp", display: "val-temp", dec: 1 },
    { id: "input-rh", display: "val-rh", dec: 0 },
    { id: "input-ws", display: "val-ws", dec: 0 },
    { id: "input-rain", display: "val-rain", dec: 1 },
    { id: "input-ffmc", display: "val-ffmc", dec: 1 },
    { id: "input-dmc", display: "val-dmc", dec: 1 },
    { id: "input-dc", display: "val-dc", dec: 1 },
    { id: "input-isi", display: "val-isi", dec: 1 },
    { id: "input-bui", display: "val-bui", dec: 1 },
    { id: "input-fwi", display: "val-fwi", dec: 1 },
  ];

  sliderConfig.forEach(({ id, display, dec }) => {
    const el = document.getElementById(id);
    const label = document.getElementById(display);
    if (el && label) {
      el.addEventListener("input", (e) => {
        label.textContent = parseFloat(e.target.value).toFixed(dec);
      });
    }
  });

  // Presets
  const presets = {
    hot: {
      Temperature: 38.0,
      RH: 30.0,
      Ws: 19.0,
      Rain: 0.0,
      FFMC: 92.5,
      DMC: 28.4,
      DC: 75.0,
      ISI: 11.2,
      BUI: 30.5,
      FWI: 18.4,
      Region: "Bejaia",
    },
    mild: {
      Temperature: 30.0,
      RH: 58.0,
      Ws: 14.0,
      Rain: 0.0,
      FFMC: 81.0,
      DMC: 11.5,
      DC: 38.0,
      ISI: 4.2,
      BUI: 13.5,
      FWI: 6.2,
      Region: "Bejaia",
    },
    safe: {
      Temperature: 23.0,
      RH: 86.0,
      Ws: 15.0,
      Rain: 7.5,
      FFMC: 46.0,
      DMC: 2.2,
      DC: 8.0,
      ISI: 0.4,
      BUI: 2.8,
      FWI: 0.2,
      Region: "Sidi-Bel Abbes",
    },
  };

  function applyPreset(data) {
    for (const [key, val] of Object.entries(data)) {
      if (key === "Region") {
        const sel = document.getElementById("input-region");
        if (sel) sel.value = val;
      } else {
        const input = document.getElementById(`input-${key.toLowerCase()}`);
        const display = document.getElementById(`val-${key.toLowerCase()}`);
        if (input) {
          input.value = val;
          if (display) display.textContent = val.toFixed(1);
        }
      }
    }
    submitPrediction();
  }

  document.getElementById("preset-hot")?.addEventListener("click", () => applyPreset(presets.hot));
  document.getElementById("preset-mild")?.addEventListener("click", () => applyPreset(presets.mild));
  document.getElementById("preset-safe")?.addEventListener("click", () => applyPreset(presets.safe));

  window.renderPredictionResult = function(pred, inputs) {
    if (!resultPlaceholder || !resultContent) return;
    resultPlaceholder.classList.add("hidden");
    resultContent.classList.remove("hidden");

    const isFire = pred.prediction_code === 1;

    riskBadge.className = `risk-badge ${isFire ? "fire" : "safe"}`;
    badgeIcon.textContent = isFire ? "🔥" : "🌿";
    riskTitle.textContent = isFire ? "HIGH WILDFIRE RISK" : "LOW RISK / SAFE CONDITIONS";
    riskDesc.textContent = isFire 
      ? "Weather indices indicate severe wildfire ignition and spread conditions." 
      : "Moisture levels and low Fire Weather Index indicate safe conditions.";

    const pFire = (pred.probability_fire * 100).toFixed(1);
    const pSafe = (pred.probability_not_fire * 100).toFixed(1);
    const conf = (pred.confidence * 100).toFixed(1);

    confidencePct.textContent = `${conf}%`;
    progressBarFill.style.width = `${conf}%`;
    progressBarFill.style.background = isFire 
      ? "linear-gradient(90deg, #ff7043, #ff5722)" 
      : "linear-gradient(90deg, #69f0ae, #00e676)";

    probFire.textContent = `${pFire}%`;
    probNotFire.textContent = `${pSafe}%`;

    // Populate Area & Location Prediction
    const areaName = document.getElementById("area-name");
    const areaSector = document.getElementById("area-sector");
    const burnedAreaVal = document.getElementById("burned-area-val");
    const spreadRadiusVal = document.getElementById("spread-radius-val");
    const spreadVelocityVal = document.getElementById("spread-velocity-val");
    const containmentVal = document.getElementById("containment-val");
    const severityBanner = document.getElementById("severity-banner");
    const radarCoords = document.getElementById("radar-coords");
    const radarEpicenter = document.getElementById("radar-epicenter");
    const radarFireDot = document.getElementById("radar-fire-dot");

    if (areaName && pred.predicted_area_name) {
      areaName.textContent = pred.predicted_area_name;
    }
    if (areaSector && pred.target_forest_sector) {
      areaSector.textContent = pred.target_forest_sector;
    }
    if (burnedAreaVal && pred.estimated_burned_area_ha !== undefined) {
      burnedAreaVal.textContent = isFire ? `${pred.estimated_burned_area_ha} ha` : "0.0 ha";
    }
    if (spreadRadiusVal && pred.spread_radius_km !== undefined) {
      spreadRadiusVal.textContent = isFire ? `${pred.spread_radius_km} km` : "0.0 km";
    }
    if (spreadVelocityVal && pred.spread_velocity_kmh !== undefined) {
      spreadVelocityVal.textContent = isFire ? `${pred.spread_velocity_kmh} km/h` : "0.0 km/h";
    }
    if (containmentVal && pred.recommended_containment_radius_km !== undefined) {
      containmentVal.textContent = isFire ? `${pred.recommended_containment_radius_km} km` : "Safe (0.0 km)";
    }
    if (severityBanner && pred.burn_severity_tier) {
      severityBanner.textContent = pred.burn_severity_tier;
      severityBanner.className = `severity-banner ${isFire ? "fire" : "safe"}`;
    }

    // Dynamic coordinates and radar position
    if (pred.spatial_coordinates && radarCoords && radarEpicenter) {
      const lat = pred.spatial_coordinates.lat;
      const lon = pred.spatial_coordinates.lon;
      const lonDir = lon >= 0 ? "E" : "W";
      radarCoords.textContent = `${lat.toFixed(2)}° N, ${Math.abs(lon).toFixed(2)}° ${lonDir}`;

      if (isFire) {
        radarFireDot.textContent = "🔥";
        radarEpicenter.style.display = "flex";
        // Position epicenter according to predicted sector
        if (pred.target_forest_sector?.includes("Sector 1") || pred.target_forest_sector?.includes("North-East")) {
          radarEpicenter.style.top = "28%";
          radarEpicenter.style.left = "68%";
        } else if (pred.target_forest_sector?.includes("Sector 3") || pred.target_forest_sector?.includes("Southern")) {
          radarEpicenter.style.top = "72%";
          radarEpicenter.style.left = "42%";
        } else if (pred.target_forest_sector?.includes("Alpha") || pred.target_forest_sector?.includes("North-West")) {
          radarEpicenter.style.top = "30%";
          radarEpicenter.style.left = "32%";
        } else {
          radarEpicenter.style.top = "50%";
          radarEpicenter.style.left = "50%";
        }
      } else {
        radarFireDot.textContent = "🛡️";
        radarEpicenter.style.top = "50%";
        radarEpicenter.style.left = "50%";
      }
    }

    // Populate Indian State Empirical Wildfire Baseline widget
    const indiaCard = document.getElementById("india-intel-card");
    if (indiaCard) {
      if (pred.is_indian_state) {
        indiaCard.classList.remove("hidden");
        const stateTitle = document.getElementById("india-intel-title");
        const stateTier = document.getElementById("india-intel-tier");
        const stateScore = document.getElementById("india-intel-score");
        const stateFill = document.getElementById("india-intel-fill");
        const statePeak = document.getElementById("india-intel-peak");
        const stateAvg = document.getElementById("india-intel-avg");
        const stateY10 = document.getElementById("india-intel-y10");
        const stateY09 = document.getElementById("india-intel-y09");
        const stateY08 = document.getElementById("india-intel-y08");

        const resolvedState = pred.state_name || inputs.Region || "State";
        if (stateTitle) stateTitle.textContent = `${resolvedState} Wildfire Baseline`;
        if (stateTier) stateTier.textContent = pred.state_risk_tier || "Monitored Zone";
        if (stateScore) stateScore.textContent = `${(pred.vulnerability_score || 0).toFixed(1)} / 100`;
        if (stateFill) stateFill.style.width = `${Math.min(100, Math.max(5, pred.vulnerability_score || 0))}%`;
        if (statePeak) statePeak.textContent = (pred.peak_annual_fires || 0).toLocaleString();
        if (stateAvg) stateAvg.textContent = `${(pred.historical_annual_avg || 0).toFixed(1)} / yr`;
        if (stateY10) stateY10.textContent = (pred.f_2010_2011 || 0).toLocaleString();
        if (stateY09) stateY09.textContent = (pred.f_2009_2010 || 0).toLocaleString();
        if (stateY08) stateY08.textContent = (pred.f_2008_2009 || 0).toLocaleString();
      } else {
        indiaCard.classList.add("hidden");
      }
    }

    factorsList.innerHTML = `
      <li><span>Noon Temperature</span><strong>${inputs.Temperature}°C ${inputs.Temperature >= 35 ? "⚠️ Elevated" : "Normal"}</strong></li>
      <li><span>Relative Humidity</span><strong>${inputs.RH}% ${inputs.RH <= 45 ? "⚠️ Critically Dry" : "Adequate"}</strong></li>
      <li><span>Fine Fuel Moisture (FFMC)</span><strong>${inputs.FFMC} ${inputs.FFMC >= 80 ? "⚠️ Highly Flammable" : "Moist"}</strong></li>
      <li><span>Fire Weather Index (FWI)</span><strong>${inputs.FWI} ${inputs.FWI >= 10 ? "⚠️ High Danger" : "Low Danger"}</strong></li>
      <li><span>Daily Rainfall</span><strong>${inputs.Rain} mm ${inputs.Rain > 0 ? "🌧️ Rain Dampening" : "0 mm Dry"}</strong></li>
    `;
  };

  async function submitPrediction(e) {
    if (e) {
      e.preventDefault();
      e.stopPropagation();
    }

    const payload = {
      Temperature: parseFloat(document.getElementById("input-temp").value),
      RH: parseFloat(document.getElementById("input-rh").value),
      Ws: parseFloat(document.getElementById("input-ws").value),
      Rain: parseFloat(document.getElementById("input-rain").value),
      FFMC: parseFloat(document.getElementById("input-ffmc").value),
      DMC: parseFloat(document.getElementById("input-dmc").value),
      DC: parseFloat(document.getElementById("input-dc").value),
      ISI: parseFloat(document.getElementById("input-isi").value),
      BUI: parseFloat(document.getElementById("input-bui").value),
      FWI: parseFloat(document.getElementById("input-fwi").value),
      Region: document.getElementById("input-region").value,
    };

    if (predictBtn) {
      predictBtn.disabled = true;
      predictBtn.innerHTML = "<span>⏳</span> Analyzing Weather Indices...";
    }

    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        throw new Error(`Server returned HTTP ${resp.status}`);
      }

      const data = await resp.json();
      window.renderPredictionResult(data.prediction, payload);

      // On tablet/mobile/narrow viewports, automatically scroll right into the grand results card
      if (window.innerWidth <= 1080) {
        setTimeout(() => {
          document.getElementById("result-container")?.scrollIntoView({ behavior: "smooth", block: "start" });
        }, 120);
      }
    } catch (err) {
      console.error("Prediction error:", err);
      alert(`Prediction Failed: ${err.message}`);
    } finally {
      if (predictBtn) {
        predictBtn.disabled = false;
        predictBtn.innerHTML = "<span class=\"btn-icon\">⚡</span> Calculate Wildfire Risk";
      }
    }
  }

  if (form) {
    form.addEventListener("submit", submitPrediction);
  }

  // Tabs
  navBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");

      navBtns.forEach((b) => b.classList.remove("active"));
      tabPanels.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      document.getElementById(targetTab)?.classList.add("active");

      if (targetTab === "tab-india") loadIndiaIntelligence();
      if (targetTab === "tab-benchmarks") loadBenchmarkMetrics();
      if (targetTab === "tab-plots") loadPlotsGallery();
    });
  });

  // Handle URL hash on load
  const hash = window.location.hash;
  if (hash === "#india") {
    document.querySelector('[data-tab="tab-india"]')?.click();
  } else if (hash === "#benchmarks") {
    document.querySelector('[data-tab="tab-benchmarks"]')?.click();
  } else if (hash === "#visualizations" || hash === "#plots") {
    document.querySelector('[data-tab="tab-plots"]')?.click();
  }

  // ================= Indian Wildfire Intelligence Loader =================
  let indiaLoaded = false;
  let allIndiaStates = [];
  let currentTierFilter = "all";
  let currentSearchQuery = "";

  async function loadIndiaIntelligence() {
    if (indiaLoaded) return;
    const tbody = document.getElementById("india-tbody");
    if (!tbody) return;

    try {
      // 1. Fetch National Summary KPI
      const sumResp = await fetch("/api/india/summary");
      if (sumResp.ok) {
        const sumData = await sumResp.json();
        const s = sumData.summary;
        if (s) {
          const kpiTotal = document.getElementById("kpi-total-fires");
          const kpiTop = document.getElementById("kpi-top-state");
          const kpiStates = document.getElementById("kpi-states-count");
          if (kpiTotal) kpiTotal.textContent = `${s.national_total_3yr_incidents.toLocaleString()} fires`;
          if (kpiTop) kpiTop.textContent = `${s.highest_fire_prone_state.name} (${s.highest_fire_prone_state.peak_fires.toLocaleString()})`;
          if (kpiStates) kpiStates.textContent = `${s.total_states_monitored} States & UTs`;
        }
      }

      // 2. Fetch All 35 States
      const statesResp = await fetch("/api/india/states");
      if (!statesResp.ok) throw new Error("Failed to fetch states dataset");
      const statesData = await statesResp.json();
      allIndiaStates = statesData.states || [];

      renderIndiaTable();
      setupIndiaControls();
      indiaLoaded = true;
    } catch (err) {
      console.error("Error loading India dataset:", err);
      if (tbody) {
        tbody.innerHTML = `<tr><td colspan="10" style="color: #ff5722;">Error loading Indian dataset: ${err.message}</td></tr>`;
      }
    }
  }

  function renderIndiaTable() {
    const tbody = document.getElementById("india-tbody");
    if (!tbody) return;

    let filtered = allIndiaStates.filter((row) => {
      // Tier filter
      const matchesTier = currentTierFilter === "all" || row["Risk_Tier"].toLowerCase().includes(currentTierFilter.toLowerCase());
      // Search filter
      const matchesSearch = !currentSearchQuery || row["States/UTs"].toLowerCase().includes(currentSearchQuery.toLowerCase());
      return matchesTier && matchesSearch;
    });

    if (filtered.length === 0) {
      tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; padding: 2rem; color: var(--text-muted);">No Indian states match your search query.</td></tr>`;
      return;
    }

    tbody.innerHTML = "";
    filtered.forEach((row, idx) => {
      const tr = document.createElement("tr");

      // Tier classification class
      let tierClass = "low";
      if (row["Risk_Tier"].includes("Extreme")) tierClass = "extreme";
      else if (row["Risk_Tier"].includes("High")) tierClass = "high";
      else if (row["Risk_Tier"].includes("Moderate")) tierClass = "moderate";

      // Rank badge style
      let rankBadgeClass = "";
      if (row["Rank"] === 1) rankBadgeClass = "rank-top-1";
      else if (row["Rank"] === 2) rankBadgeClass = "rank-top-2";
      else if (row["Rank"] === 3) rankBadgeClass = "rank-top-3";

      const vulnScore = (row["Vulnerability_Score"] || 0).toFixed(1);

      tr.innerHTML = `
        <td><span class="state-rank-badge ${rankBadgeClass}">${row["Rank"] || (idx + 1)}</span></td>
        <td><strong>${row["States/UTs"]}</strong></td>
        <td>${row["2010-2011"].toLocaleString()}</td>
        <td>${row["2009-10"].toLocaleString()}</td>
        <td>${row["2008-09"].toLocaleString()}</td>
        <td><strong>${row["3Year_Average"].toLocaleString()}</strong></td>
        <td><span style="color: #ff7043; font-weight: 600;">${row["Peak_Annual_Fires"].toLocaleString()}</span></td>
        <td>
          <div class="table-vuln-bar">
            <div class="table-vuln-track">
              <div class="table-vuln-fill" style="width: ${Math.min(100, Math.max(4, vulnScore))}%;"></div>
            </div>
            <span>${vulnScore}</span>
          </div>
        </td>
        <td><span class="tier-tag ${tierClass}">${row["Risk_Tier"]}</span></td>
        <td style="text-align: center;">
          <button type="button" class="table-action-btn" data-state="${row["States/UTs"]}">
            ⚡ Simulate
          </button>
        </td>
      `;

      // Wire action button
      const actionBtn = tr.querySelector(".table-action-btn");
      actionBtn?.addEventListener("click", () => {
        simulateForState(row["States/UTs"]);
      });

      tbody.appendChild(tr);
    });
  }

  function simulateForState(stateName) {
    // Switch to predict tab
    document.querySelector('[data-tab="tab-predict"]')?.click();

    // Select state in dropdown
    const regionSelect = document.getElementById("input-region");
    if (regionSelect) {
      // Find matching option
      for (let i = 0; i < regionSelect.options.length; i++) {
        if (regionSelect.options[i].value.toLowerCase() === stateName.toLowerCase()) {
          regionSelect.selectedIndex = i;
          break;
        }
      }
    }

    // Trigger calculation
    submitPrediction();

    // Smooth scroll to results
    setTimeout(() => {
      document.getElementById("result-container")?.scrollIntoView({ behavior: "smooth" });
    }, 200);
  }

  function setupIndiaControls() {
    // Search input
    const searchInput = document.getElementById("india-search-input");
    if (searchInput) {
      searchInput.addEventListener("input", (e) => {
        currentSearchQuery = e.target.value.trim();
        renderIndiaTable();
      });
    }

    // Tier filter pills
    const pills = document.querySelectorAll("#tier-filter-pills .tier-pill");
    pills.forEach((pill) => {
      pill.addEventListener("click", () => {
        pills.forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");
        currentTierFilter = pill.getAttribute("data-filter") || "all";
        renderIndiaTable();
      });
    });

    // Chart preview lightbox
    const chartCard = document.getElementById("india-chart-preview");
    if (chartCard && lightboxImg && lightbox) {
      chartCard.addEventListener("click", () => {
        lightboxImg.src = "/plots/india_state_fire_incident_rankings.png";
        if (lightboxCaption) lightboxCaption.textContent = "Annual Average Forest Fire Incidents Across Indian States (2008–2011)";
        lightbox.classList.add("active");
      });
    }
  }

  // Benchmark Loader
  let metricsLoaded = false;
  async function loadBenchmarkMetrics() {
    if (metricsLoaded) return;
    const tbody = document.getElementById("benchmark-tbody");
    if (!tbody) return;

    try {
      const resp = await fetch("/api/metrics");
      const data = await resp.json();
      tbody.innerHTML = "";

      data.models.forEach((row, idx) => {
        const isWinner = idx === 0;
        const tr = document.createElement("tr");
        if (isWinner) tr.classList.add("winner-row");

        tr.innerHTML = `
          <td><strong>${row["Model"]}</strong></td>
          <td>${row["CV Mean F1 (Train)"] !== undefined ? row["CV Mean F1 (Train)"].toFixed(4) : "-"}</td>
          <td>${row["CV Std F1"] !== undefined ? "±" + row["CV Std F1"].toFixed(4) : "-"}</td>
          <td>${row["CV Mean ROC-AUC"] !== undefined ? row["CV Mean ROC-AUC"].toFixed(4) : "-"}</td>
          <td>${(row["Test Accuracy"] * 100).toFixed(2)}%</td>
          <td>${(row["Test Precision"] * 100).toFixed(2)}%</td>
          <td>${(row["Test Recall"] * 100).toFixed(2)}%</td>
          <td><strong>${row["Test F1-Score"]?.toFixed(4) || "-"}</strong></td>
          <td><strong>${row["Test ROC-AUC"]?.toFixed(4) || "-"}</strong></td>
        `;
        tbody.appendChild(tr);
      });
      metricsLoaded = true;
    } catch (err) {
      tbody.innerHTML = `<tr><td colspan="9" style="color: #ff5722;">Error loading metrics: ${err.message}</td></tr>`;
    }
  }

  // Plots Gallery Loader
  let plotsLoaded = false;
  const lightbox = document.getElementById("lightbox");
  const lightboxImg = document.getElementById("lightbox-img");
  const lightboxCaption = document.getElementById("lightbox-caption");
  const lightboxClose = document.getElementById("lightbox-close");

  async function loadPlotsGallery() {
    if (plotsLoaded) return;
    const gallery = document.getElementById("plots-grid");
    if (!gallery) return;

    try {
      const resp = await fetch("/api/plots");
      const data = await resp.json();
      gallery.innerHTML = "";

      data.plots.forEach((plot) => {
        const card = document.createElement("div");
        card.className = "plot-card";
        card.innerHTML = `
          <img class="plot-thumb" src="${plot.url}" alt="${plot.name}" loading="lazy">
          <div class="plot-info">
            <h4 class="plot-title">${plot.name}</h4>
          </div>
        `;
        card.addEventListener("click", () => {
          lightboxImg.src = plot.url;
          lightboxCaption.textContent = plot.name;
          lightbox.classList.add("active");
        });
        gallery.appendChild(card);
      });
      plotsLoaded = true;
    } catch (err) {
      gallery.innerHTML = `<p style="color: #ff5722;">Failed to load plots: ${err.message}</p>`;
    }
  }

  lightboxClose?.addEventListener("click", () => lightbox.classList.remove("active"));
  lightbox?.addEventListener("click", (e) => {
    if (e.target === lightbox) lightbox.classList.remove("active");
  });
}

// Ensure execution whether DOM is already loaded or loading
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initDashboard);
} else {
  initDashboard();
}
