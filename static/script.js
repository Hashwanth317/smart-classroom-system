/* Smart Classroom Real-Time Dashboard Client Logic
   Manages automated polling, mode switching, search filtering, and hardware telemetry.
*/

let currentMode = "ENTRY";
let cachedAttendance = [];
let dashboardStreamActive = false;
let attendanceProcessingInterval = null;
const attendanceCanvas = document.createElement("canvas");
attendanceCanvas.width = 640;
attendanceCanvas.height = 480;
const attendanceCtx = attendanceCanvas.getContext("2d");

document.addEventListener("DOMContentLoaded", () => {
  // Initial fetch and start periodic polling
  fetchDashboardData();
  setInterval(fetchDashboardData, 1500);

  // Initialize laptop webcam in browser
  initDashboardWebcam();
});

async function initDashboardWebcam() {
  const video = document.getElementById("liveWebcamVideo");
  const overlay = document.getElementById("attendanceOverlayCanvas");
  const promptOverlay = document.getElementById("camPromptOverlay");
  const fallbackImg = document.getElementById("liveVideoFeed");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" }
    });
    if (video) {
      video.srcObject = stream;
      video.style.display = "block";
      video.play();
    }
    if (overlay) overlay.style.display = "block";
    if (promptOverlay) promptOverlay.style.display = "none";
    if (fallbackImg) fallbackImg.style.display = "none";
    dashboardStreamActive = true;

    if (video) {
      video.onloadedmetadata = () => {
        if (overlay) {
          overlay.width = video.videoWidth || 640;
          overlay.height = video.videoHeight || 480;
        }
      };
    }

    // Start background frame recognition loop
    if (attendanceProcessingInterval) clearInterval(attendanceProcessingInterval);
    attendanceProcessingInterval = setInterval(sendAttendanceFrame, 220);

  } catch (err) {
    console.warn("Direct getUserMedia fallback to server stream:", err);
    if (video) video.style.display = "none";
    if (overlay) overlay.style.display = "none";
    if (fallbackImg) fallbackImg.style.display = "block";
    if (promptOverlay && (err.name === "NotAllowedError" || err.name === "PermissionDeniedError")) {
      promptOverlay.style.display = "flex";
    }
  }
}

async function sendAttendanceFrame() {
  const video = document.getElementById("liveWebcamVideo");
  const overlay = document.getElementById("attendanceOverlayCanvas");
  if (!dashboardStreamActive || !video || video.videoWidth === 0) return;

  attendanceCtx.drawImage(video, 0, 0, 640, 480);
  const dataUrl = attendanceCanvas.toDataURL("image/jpeg", 0.78);

  try {
    const res = await fetch("/api/attendance/process_frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl })
    });
    const data = await res.json();
    if (data.success && data.annotated_frame) {
      // Draw annotated overlay on canvas
      const img = new Image();
      img.onload = () => {
        if (overlay) {
          const ctx = overlay.getContext("2d");
          ctx.clearRect(0, 0, overlay.width, overlay.height);
          ctx.drawImage(img, 0, 0, overlay.width, overlay.height);
        }
      };
      img.src = data.annotated_frame;
    }
  } catch (err) {
    // Non-blocking frame drop
  }
}

async function fetchDashboardData() {
  try {
    const res = await fetch("/api/dashboard");
    if (!res.ok) return;
    const data = await res.json();
    if (!data.success) return;

    updateMetrics(data.metrics);
    updateActivityFeed(data.activity);
    
    // Cache attendance for instant client-side search filtering
    cachedAttendance = data.attendance || [];
    renderAttendanceTable(cachedAttendance);

  } catch (err) {
    console.error("Dashboard poll error:", err);
  }
}

function updateMetrics(metrics) {
  if (!metrics) return;

  // 1. Students Present
  const presentEl = document.getElementById("metricPresentCount");
  if (presentEl) presentEl.textContent = metrics.students_present ?? 0;

  // 2. Current Occupancy
  const occEl = document.getElementById("metricOccupancy");
  if (occEl) occEl.textContent = metrics.occupancy ?? 0;

  // 3. Classroom Status
  const classBadge = document.getElementById("metricClassroomBadge");
  if (classBadge) {
    const isOccupied = metrics.classroom_status === "OCCUPIED";
    classBadge.textContent = metrics.classroom_status || "VACANT";
    classBadge.className = isOccupied ? "status-badge-occupied" : "status-badge-vacant";
  }

  // 4. Light Status
  const lightEl = document.getElementById("metricLightStatus");
  const lightDot = document.getElementById("lightDot");
  if (lightEl) lightEl.textContent = metrics.light_status || "OFF";
  if (lightDot) {
    lightDot.className = metrics.light_status === "ON" ? "indicator-dot indicator-on" : "indicator-dot indicator-off";
  }

  // 5. Fan Status
  const fanEl = document.getElementById("metricFanStatus");
  const fanDot = document.getElementById("fanDot");
  if (fanEl) fanEl.textContent = metrics.fan_status || "OFF";
  if (fanDot) {
    fanDot.className = metrics.fan_status === "ON" ? "indicator-dot indicator-on" : "indicator-dot indicator-off";
  }

  // 6. Camera Status
  const camEl = document.getElementById("metricCameraStatus");
  const camDot = document.getElementById("cameraDot");
  const camBadge = document.getElementById("cameraBadge");
  const isCamOnline = metrics.camera_status === "ONLINE";

  if (camEl) camEl.textContent = metrics.camera_status || "OFFLINE";
  if (camDot) {
    camDot.className = isCamOnline ? "indicator-dot indicator-on" : "indicator-dot indicator-off";
  }
  if (camBadge) {
    camBadge.className = isCamOnline ? "camera-status-pill camera-online" : "camera-status-pill camera-offline";
    camBadge.innerHTML = `<span class="indicator-dot ${isCamOnline ? 'indicator-on' : 'indicator-off'}"></span> CAMERA ${metrics.camera_status}`;
  }

  // 7. Face Recognition Status
  const faceEl = document.getElementById("metricFaceStatus");
  if (faceEl) faceEl.textContent = metrics.face_recognition_status || "STANDBY";

  // 8. Attendance Mode Buttons Sync
  currentMode = metrics.attendance_mode || "ENTRY";
  syncModeUI(currentMode);
}

function syncModeUI(mode) {
  const badge = document.getElementById("currentModeBadge");
  const btnEntry = document.getElementById("btnEntryMode");
  const btnExit = document.getElementById("btnExitMode");

  if (badge) {
    badge.textContent = `${mode} MODE`;
    badge.className = mode === "ENTRY" ? "mode-badge-large mode-entry-badge" : "mode-badge-large mode-exit-badge";
  }

  if (btnEntry && btnExit) {
    if (mode === "ENTRY") {
      btnEntry.className = "mode-btn active entry";
      btnExit.className = "mode-btn exit";
    } else {
      btnEntry.className = "mode-btn entry";
      btnExit.className = "mode-btn active exit";
    }
  }
}

async function switchMode(mode) {
  try {
    const res = await fetch("/api/attendance/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: mode })
    });
    const data = await res.json();
    if (data.success) {
      currentMode = data.current_mode;
      syncModeUI(currentMode);
    }
  } catch (err) {
    console.error("Error switching mode:", err);
  }
}

async function startCamera() {
  try {
    await fetch("/api/camera/start", { method: "POST" });
    const img = document.getElementById("liveVideoFeed");
    if (img) img.src = "/video_feed?" + new Date().getTime();
  } catch (err) {
    console.error("Error starting camera:", err);
  }
}

async function stopCamera() {
  try {
    await fetch("/api/camera/stop", { method: "POST" });
  } catch (err) {
    console.error("Error stopping camera:", err);
  }
}

function updateActivityFeed(activities) {
  const listEl = document.getElementById("activityList");
  if (!listEl) return;

  if (!activities || activities.length === 0) {
    listEl.innerHTML = `
      <li class="activity-item">
        <span class="activity-time">--:--</span>
        <span class="activity-msg">No events recorded yet today.</span>
      </li>
    `;
    return;
  }

  listEl.innerHTML = "";
  activities.forEach(item => {
    const li = document.createElement("li");
    let typeClass = item.event_type || "";
    li.className = `activity-item ${typeClass}`;

    // Extract formatted time
    let timeStr = item.timestamp || "";
    if (timeStr.includes(" ")) {
      timeStr = timeStr.split(" ")[1];
    }

    li.innerHTML = `
      <span class="activity-time">${timeStr}</span>
      <span class="activity-msg">${escapeHtml(item.message)}</span>
    `;
    listEl.appendChild(li);
  });
}

function renderAttendanceTable(records) {
  const tbody = document.getElementById("attendanceTableBody");
  if (!tbody) return;

  const rollFilter = (document.getElementById("searchRollNo")?.value || "").trim().toUpperCase();
  const nameFilter = (document.getElementById("searchName")?.value || "").trim().toUpperCase();

  const filtered = records.filter(r => {
    const matchRoll = !rollFilter || (r.roll_no && r.roll_no.toUpperCase().includes(rollFilter));
    const matchName = !nameFilter || (r.name && r.name.toUpperCase().includes(nameFilter));
    return matchRoll && matchName;
  });

  if (filtered.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 2rem;">
          No matching student attendance records found.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = "";
  filtered.forEach(r => {
    const tr = document.createElement("tr");

    let statusBadge = "";
    if (r.status === "Present") {
      statusBadge = '<span class="badge badge-present">Present</span>';
    } else if (r.status === "Left") {
      statusBadge = '<span class="badge badge-left">Left</span>';
    } else {
      statusBadge = '<span class="badge badge-not-entered">Not Entered</span>';
    }

    tr.innerHTML = `
      <td style="font-weight: 700; color: var(--primary);">${escapeHtml(r.roll_no)}</td>
      <td style="font-weight: 600;">${escapeHtml(r.name)}</td>
      <td>${r.entry_time || '<span style="color: var(--text-muted);">&mdash;</span>'}</td>
      <td>${r.exit_time || '<span style="color: var(--text-muted);">&mdash;</span>'}</td>
      <td>${statusBadge}</td>
    `;
    tbody.appendChild(tr);
  });
}

function filterAttendance() {
  renderAttendanceTable(cachedAttendance);
}

async function sendHardwareTelemetry() {
  const occ = parseInt(document.getElementById("simOccupancy").value) || 0;
  const light = document.getElementById("simLight").value === "true";
  const fan = document.getElementById("simFan").value === "true";
  const pir = document.getElementById("simPIR").value === "true";
  const statusEl = document.getElementById("telemetryStatus");

  statusEl.textContent = "Sending...";
  statusEl.style.color = "var(--primary)";

  try {
    const res = await fetch("/api/hardware/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        occupancy: occ,
        lightStatus: light,
        fanStatus: fan,
        pirStatus: pir
      })
    });
    const data = await res.json();
    if (data.success) {
      statusEl.textContent = "Telemetry Synced!";
      statusEl.style.color = "var(--success)";
      fetchDashboardData();
      setTimeout(() => { statusEl.textContent = ""; }, 3000);
    } else {
      statusEl.textContent = "Error updating hardware";
      statusEl.style.color = "var(--danger)";
    }
  } catch (err) {
    statusEl.textContent = "Network Error";
    statusEl.style.color = "var(--danger)";
  }
}

function escapeHtml(text) {
  if (!text) return "";
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
