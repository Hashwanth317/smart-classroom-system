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

  // 2. Current Occupancy (Room Count)
  const occEl = document.getElementById("metricOccupancy");
  if (occEl) occEl.textContent = metrics.occupancy ?? 0;

  // 2b. IN & OUT Counts (IR Sensors)
  const inEl = document.getElementById("metricInCount");
  if (inEl) inEl.textContent = metrics.in_count ?? 0;

  const outEl = document.getElementById("metricOutCount");
  if (outEl) outEl.textContent = metrics.out_count ?? 0;

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
        <td colspan="6" style="text-align: center; color: var(--text-muted); padding: 2rem;">
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
      <td style="font-weight: 700; color: var(--primary); cursor: pointer;" onclick="openStudentAttendanceModal('${escapeHtml(r.roll_no)}')" title="Click to view attendance history">${escapeHtml(r.roll_no)}</td>
      <td style="font-weight: 600; cursor: pointer;" onclick="openStudentAttendanceModal('${escapeHtml(r.roll_no)}')" title="Click to view attendance history">${escapeHtml(r.name)}</td>
      <td>${r.entry_time || '<span style="color: var(--text-muted);">&mdash;</span>'}</td>
      <td>${r.exit_time || '<span style="color: var(--text-muted);">&mdash;</span>'}</td>
      <td>${statusBadge}</td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick="openStudentAttendanceModal('${escapeHtml(r.roll_no)}')" title="View Attendance History">
          <svg width="13" height="13" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" style="vertical-align:-1px; margin-right:3px;">
            <path stroke-linecap="round" stroke-linejoin="round" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
          </svg>
          View Attendance
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function filterAttendance() {
  renderAttendanceTable(cachedAttendance);
}

function autoCalcSimulator() {
  const inVal = parseInt(document.getElementById("simInCount")?.value) || 0;
  const outVal = parseInt(document.getElementById("simOutCount")?.value) || 0;
  const room = Math.max(0, inVal - outVal);

  const roomEl = document.getElementById("simRoomCount");
  if (roomEl) roomEl.value = room;

  const lightBadge = document.getElementById("simLightBadge");
  const fanBadge = document.getElementById("simFanBadge");

  const lightOn = (room > 0);
  const fanOn = (room > 5);

  if (lightBadge) {
    lightBadge.textContent = lightOn ? "ON" : "OFF";
    lightBadge.className = lightOn ? "badge badge-present" : "badge badge-not-entered";
  }
  if (fanBadge) {
    fanBadge.textContent = fanOn ? "ON" : "OFF";
    fanBadge.className = fanOn ? "badge badge-present" : "badge badge-not-entered";
  }
}

async function sendHardwareTelemetry() {
  const inVal = parseInt(document.getElementById("simInCount")?.value) || 0;
  const outVal = parseInt(document.getElementById("simOutCount")?.value) || 0;
  const roomVal = Math.max(0, inVal - outVal);
  const light = (roomVal > 0);
  const fan = (roomVal > 5);

  const statusEl = document.getElementById("telemetryStatus");
  statusEl.textContent = "Sending...";
  statusEl.style.color = "var(--primary)";

  try {
    const res = await fetch("/api/hardware/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        inCount: inVal,
        outCount: outVal,
        roomCount: roomVal,
        lightStatus: light,
        fanStatus: fan
      })
    });
    const data = await res.json();
    if (data.success) {
      statusEl.textContent = "IR Telemetry Synced!";
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

let currentModalRollNo = null;

async function openStudentAttendanceModal(rollNo) {
  currentModalRollNo = rollNo;
  const modal = document.getElementById("attendanceModal");
  if (!modal) return;
  const title = document.getElementById("modalStudentTitle");
  const tbody = document.getElementById("modalAttendanceTableBody");
  const statPresent = document.getElementById("modalPresentCount");
  const statTotal = document.getElementById("modalTotalDays");
  const statRate = document.getElementById("modalAttendanceRate");

  modal.classList.add("active");
  title.innerHTML = `
    <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" style="color:var(--primary);">
      <path stroke-linecap="round" stroke-linejoin="round" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
    </svg>
    Attendance History: ${escapeHtml(rollNo)}
  `;
  tbody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding:1.5rem; color:var(--text-muted);">Fetching attendance records...</td></tr>`;

  try {
    const res = await fetch(`/api/students/${encodeURIComponent(rollNo)}/attendance`);
    const data = await res.json();

    if (!data.success) {
      tbody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding:1.5rem; color:var(--danger);">${data.error || 'Failed to load records'}</td></tr>`;
      return;
    }

    title.innerHTML = `
      <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" stroke-width="2" style="color:var(--primary);">
        <path stroke-linecap="round" stroke-linejoin="round" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
      </svg>
      ${escapeHtml(data.student.name)} (${escapeHtml(data.student.roll_no)})
    `;
    if (statPresent) statPresent.textContent = data.present_days;
    if (statTotal) statTotal.textContent = data.total_records;
    if (statRate) statRate.textContent = `${data.attendance_rate}%`;

    if (!data.records || data.records.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding:1.5rem; color:var(--text-muted);">No attendance records found for this student yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = "";
    data.records.forEach(r => {
      const tr = document.createElement("tr");
      let badge = r.status === "Present" 
        ? '<span class="badge badge-present">Present</span>'
        : (r.status === "Left" ? '<span class="badge badge-left">Left</span>' : '<span class="badge badge-not-entered">Not Entered</span>');
      tr.innerHTML = `
        <td style="font-weight:600;">${r.date}</td>
        <td>${r.entry_time || '<span style="color:var(--text-muted);">&mdash;</span>'}</td>
        <td>${r.exit_time || '<span style="color:var(--text-muted);">&mdash;</span>'}</td>
        <td>${badge}</td>
      `;
      tbody.appendChild(tr);
    });

  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding:1.5rem; color:var(--danger);">Error: ${escapeHtml(err.message)}</td></tr>`;
  }
}

function closeStudentAttendanceModal() {
  const modal = document.getElementById("attendanceModal");
  if (modal) modal.classList.remove("active");
}

async function manualRefreshActivity() {
  const btn = document.getElementById("btnRefreshActivity");
  const icon = document.getElementById("refreshIcon");
  const text = document.getElementById("refreshBtnText");
  const lastUpdated = document.getElementById("activityLastUpdated");

  if (btn) btn.disabled = true;
  if (icon) icon.classList.add("spinning");
  if (text) text.textContent = "Refreshing...";

  try {
    await fetchDashboardData();
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    if (lastUpdated) {
      lastUpdated.textContent = `Updated ${timeStr}`;
      lastUpdated.style.color = "var(--primary)";
      setTimeout(() => {
        if (lastUpdated) lastUpdated.style.color = "var(--text-muted)";
      }, 2500);
    }
  } catch (err) {
    console.error("Manual refresh error:", err);
  } finally {
    if (icon) icon.classList.remove("spinning");
    if (text) text.textContent = "Refresh";
    if (btn) btn.disabled = false;
  }
}

async function manualRefreshAttendance() {
  const btn = document.getElementById("btnRefreshAttendance");
  const icon = document.getElementById("refreshAttendanceIcon");
  const text = document.getElementById("refreshAttendanceText");
  const lastUpdated = document.getElementById("attendanceLastUpdated");

  if (btn) btn.disabled = true;
  if (icon) icon.classList.add("spinning");
  if (text) text.textContent = "Refreshing...";

  try {
    await fetchDashboardData();
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    if (lastUpdated) {
      lastUpdated.textContent = `• Updated ${timeStr}`;
      lastUpdated.style.color = "var(--primary)";
      setTimeout(() => {
        if (lastUpdated) lastUpdated.style.color = "var(--text-muted)";
      }, 2500);
    }
  } catch (err) {
    console.error("Attendance refresh error:", err);
  } finally {
    if (icon) icon.classList.remove("spinning");
    if (text) text.textContent = "Refresh Sheet";
    if (btn) btn.disabled = false;
  }
}

async function refreshStudentModalAttendance() {
  if (!currentModalRollNo) return;
  const icon = document.getElementById("refreshStudentModalIcon");
  if (icon) icon.classList.add("spinning");
  try {
    await openStudentAttendanceModal(currentModalRollNo);
  } finally {
    if (icon) icon.classList.remove("spinning");
  }
}
