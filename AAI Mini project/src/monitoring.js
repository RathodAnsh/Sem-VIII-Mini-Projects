// Import MediaPipe tasks
import * as MediaPipeVision from '@mediapipe/tasks-vision';

// MediaPipe references
let FaceLandmarker;
let FilesetResolver;

let faceLandmarker;
let webcamStream;
let animationFrameId;
let lastVideoTime = -1;
let fpsInterval;
let frameCount = 0;
let lastFpsTime = performance.now();

// Initialize MediaPipe references
async function initializeMediaPipe() {
  FaceLandmarker = MediaPipeVision.FaceLandmarker;
  FilesetResolver = MediaPipeVision.FilesetResolver;
}

// Initialize on load
initializeMediaPipe();

const video = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const canvasCtx = canvas.getContext('2d');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const videoPlaceholder = document.getElementById('video-placeholder');
const alertSound = document.getElementById('alert-sound');
const alertPanel = document.getElementById('alert-panel');
const alertMessage = document.getElementById('alert-message');

// Proper EAR (Eye Aspect Ratio) thresholds
// Lowered slightly to accommodate naturally narrower eye shapes universally
const EAR_THRESHOLD_CLOSED = 0.18;      
const EAR_THRESHOLD_PARTIALLY = 0.22;   

// Mouth thresholds for yawning (Using a simplified, highly accurate ratio)
const MAR_THRESHOLD_YAWN = 0.6;         
const MAR_THRESHOLD_ALERT = 0.8;        

// Head pose thresholds (Using Ratios instead of Degrees for scale invariance)
// A ratio of 1.0 means perfectly centered.
const YAW_RATIO_LEFT = 0.4;             // Nose is too close to left cheek
const YAW_RATIO_RIGHT = 2.5;            // Nose is too close to right cheek

const CONSECUTIVE_FRAMES = 15;          // Fast response time

let eyeClosedFrames = 0;
let eyePartiallyClosedFrames = 0;
let yawnFrames = 0;
let headTurnedFrames = 0;
let isAlertPlaying = false;
let currentAlerts = new Set();

const EYE_INDICES = {
  LEFT_EYE_TOP: 159,
  LEFT_EYE_BOTTOM: 145,
  LEFT_EYE_LEFT: 33,
  LEFT_EYE_RIGHT: 133,
  RIGHT_EYE_TOP: 386,
  RIGHT_EYE_BOTTOM: 374,
  RIGHT_EYE_LEFT: 362,
  RIGHT_EYE_RIGHT: 263
};

const MOUTH_INDICES = {
  UPPER_LIP: 13,
  LOWER_LIP: 14,
  MOUTH_LEFT: 78,
  MOUTH_RIGHT: 308
};

const HEAD_INDICES = {
  NOSE_TIP: 1,
  LEFT_EYE: 33,
  RIGHT_EYE: 263,
  CHIN: 152
};

function euclideanDistance(point1, point2) {
  const dx = point1.x - point2.x;
  const dy = point1.y - point2.y;
  const dz = point1.z - point2.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function calculateEAR(landmarks) {
  // Corrected MediaPipe landmarks for true vertical and horizontal measurements
  // Left Eye
  const leftEye_V1 = euclideanDistance(landmarks[160], landmarks[144]); // Inner vertical
  const leftEye_V2 = euclideanDistance(landmarks[158], landmarks[153]); // Outer vertical
  const leftEye_H = euclideanDistance(landmarks[33], landmarks[133]);   // Horizontal
  
  // Right Eye
  const rightEye_V1 = euclideanDistance(landmarks[385], landmarks[380]); // Inner vertical
  const rightEye_V2 = euclideanDistance(landmarks[387], landmarks[373]); // Outer vertical
  const rightEye_H = euclideanDistance(landmarks[362], landmarks[263]);  // Horizontal
  
  // EAR formula
  const leftEAR = (leftEye_V1 + leftEye_V2) / (2.0 * leftEye_H);
  const rightEAR = (rightEye_V1 + rightEye_V2) / (2.0 * rightEye_H);
  
  return (leftEAR + rightEAR) / 2.0;
}

function calculateMAR(landmarks) {
  // Simplified MAR targeting the inner lip distance vs total mouth width
  // This prevents false yawning alarms for users with naturally thicker lips
  const mouthVertical = euclideanDistance(landmarks[13], landmarks[14]);   // Upper inner lip to lower inner lip
  const mouthHorizontal = euclideanDistance(landmarks[78], landmarks[308]); // Left corner to right corner
  
  // MAR formula
  return mouthVertical / mouthHorizontal;
}

function calculateHeadPose(landmarks) {
  // Using 2D Head Pose Ratio (HPR) instead of unreliable 3D Z-axis estimation.
  // This works universally regardless of how close the driver is to the camera or their face width.
  const noseTip = landmarks[1];
  const leftCheek = landmarks[234];
  const rightCheek = landmarks[454];
  
  // Calculate horizontal distance from nose to edges of face
  const leftDist = Math.abs(noseTip.x - leftCheek.x);
  const rightDist = Math.abs(rightCheek.x - noseTip.x);
  
  // Prevent division by zero
  const safeRightDist = rightDist === 0 ? 0.001 : rightDist;
  
  // Calculate Yaw Ratio
  const yawRatio = leftDist / safeRightDist;
  
  return {
    yawRatio: yawRatio,
    // Keep pitch and roll as 0 for UI stability unless explicitly needed later
    pitch: 0, 
    roll: 0 
  };
}

function updateAlert(type, isActive) {
  if (isActive) {
    currentAlerts.add(type);
  } else {
    currentAlerts.delete(type);
  }

  if (currentAlerts.size > 0) {
    alertPanel.classList.remove('hidden');
    alertMessage.textContent = Array.from(currentAlerts).join(' | ');
    
    // Play alarm and ensure it keeps looping while alerts are active
    if (!isAlertPlaying) {
      alertSound.loop = true;
      alertSound.volume = 1.0;  // Max volume for emergency alert
      alertSound.play().catch(e => console.log('Audio play failed:', e));
      isAlertPlaying = true;
    }
  } else {
    if (isAlertPlaying) {
      alertSound.loop = false;
      alertSound.pause();
      alertSound.currentTime = 0;
      isAlertPlaying = false;
    }
    alertPanel.classList.add('hidden');
  }
}

function updateUI(ear, mar, headPose, eyeStatus, yawnStatus, headStatus, driverStatus) {
  // Display EAR, MAR values
  document.getElementById('ear-value').textContent = `EAR: ${ear.toFixed(3)}`;
  document.getElementById('mar-value').textContent = `MAR: ${mar.toFixed(3)}`;
  
  // Display head pose angles using the new Head Pose Ratio
  const headAngleText = `Yaw Ratio: ${headPose.yawRatio.toFixed(2)} | Pitch: N/A | Roll: N/A`;
  document.getElementById('head-angle').textContent = headAngleText;

  // Update eye status with proper color coding
  document.getElementById('eye-status').textContent = eyeStatus;
  if (eyeStatus === 'OPEN') {
    document.getElementById('eye-status').className = 'font-semibold text-green-400';
  } else if (eyeStatus === 'PARTIALLY_CLOSED') {
    document.getElementById('eye-status').className = 'font-semibold text-yellow-400';
  } else {
    document.getElementById('eye-status').className = 'font-semibold text-red-400';
  }

  // Update yawning status
  document.getElementById('yawn-status').textContent = yawnStatus;
  document.getElementById('yawn-status').className = yawnStatus === 'NORMAL' ? 'font-semibold text-green-400' : 'font-semibold text-yellow-400';

  // Update head status
  document.getElementById('head-status').textContent = headStatus;
  document.getElementById('head-status').className = headStatus === 'FORWARD' ? 'font-semibold text-green-400' : 'font-semibold text-yellow-400';

  const statusElement = document.getElementById('driver-status');
  const statusIndicator = document.getElementById('status-indicator');

  statusElement.textContent = driverStatus;

  // Update status based on alert conditions - EMERGENCY RED if ANY alert is active
  if (currentAlerts.size > 0) {
    statusElement.className = 'text-2xl font-bold text-red-500';
    statusIndicator.className = 'w-2 h-2 rounded-full bg-red-500 animate-pulse';
  } else if (driverStatus === 'SAFE') {
    statusElement.className = 'text-2xl font-bold text-green-400';
    statusIndicator.className = 'w-2 h-2 rounded-full bg-green-500 animate-pulse';
  } else {
    statusElement.className = 'text-2xl font-bold text-yellow-400';
    statusIndicator.className = 'w-2 h-2 rounded-full bg-yellow-500 animate-pulse';
  }
}

function drawFaceOverlay(landmarks) {
  if (!landmarks || landmarks.length === 0) return;

  let minX = 1, minY = 1, maxX = 0, maxY = 0;

  landmarks.forEach(landmark => {
    minX = Math.min(minX, landmark.x);
    minY = Math.min(minY, landmark.y);
    maxX = Math.max(maxX, landmark.x);
    maxY = Math.max(maxY, landmark.y);
  });

  const padding = 0.05;
  minX = Math.max(0, minX - padding);
  minY = Math.max(0, minY - padding);
  maxX = Math.min(1, maxX + padding);
  maxY = Math.min(1, maxY + padding);

  const x = minX * canvas.width;
  const y = minY * canvas.height;
  const width = (maxX - minX) * canvas.width;
  const height = (maxY - minY) * canvas.height;

  const isAlert = currentAlerts.size === 0;
  const color = isAlert ? '#10b981' : '#ef4444';

  canvasCtx.strokeStyle = color;
  canvasCtx.lineWidth = 3;
  canvasCtx.strokeRect(x, y, width, height);

  const cornerLength = 20;
  canvasCtx.beginPath();
  canvasCtx.moveTo(x, y + cornerLength);
  canvasCtx.lineTo(x, y);
  canvasCtx.lineTo(x + cornerLength, y);
  canvasCtx.moveTo(x + width - cornerLength, y);
  canvasCtx.lineTo(x + width, y);
  canvasCtx.lineTo(x + width, y + cornerLength);
  canvasCtx.moveTo(x + width, y + height - cornerLength);
  canvasCtx.lineTo(x + width, y + height);
  canvasCtx.lineTo(x + width - cornerLength, y + height);
  canvasCtx.moveTo(x + cornerLength, y + height);
  canvasCtx.lineTo(x, y + height);
  canvasCtx.lineTo(x, y + height - cornerLength);
  canvasCtx.lineWidth = 4;
  canvasCtx.stroke();

  const centerX = x + width / 2;
  const centerY = y + height / 2;
  const reticleSize = 15;

  canvasCtx.beginPath();
  canvasCtx.moveTo(centerX - reticleSize, centerY);
  canvasCtx.lineTo(centerX + reticleSize, centerY);
  canvasCtx.moveTo(centerX, centerY - reticleSize);
  canvasCtx.lineTo(centerX, centerY + reticleSize);
  canvasCtx.lineWidth = 2;
  canvasCtx.stroke();

  canvasCtx.beginPath();
  canvasCtx.arc(centerX, centerY, 8, 0, 2 * Math.PI);
  canvasCtx.stroke();
}

function updateFPS() {
  frameCount++;
  const currentTime = performance.now();
  const elapsed = currentTime - lastFpsTime;

  if (elapsed >= 1000) {
    const fps = Math.round((frameCount * 1000) / elapsed);
    document.getElementById('fps-counter').textContent = `FPS: ${fps}`;
    frameCount = 0;
    lastFpsTime = currentTime;
  }
}

async function predictWebcam() {
  if (!faceLandmarker) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

  const startTimeMs = performance.now();

  if (lastVideoTime !== video.currentTime) {
    lastVideoTime = video.currentTime;
    const results = faceLandmarker.detectForVideo(video, startTimeMs);

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
      const landmarks = results.faceLandmarks[0];

      const ear = calculateEAR(landmarks);
      const mar = calculateMAR(landmarks);
      const headPose = calculateHeadPose(landmarks);

      let eyeStatus = 'OPEN';
      let yawnStatus = 'NORMAL';
      let headStatus = 'FORWARD';
      let driverStatus = 'SAFE';

      // ==================== EYE STATUS DETECTION ====================
      // Properly detect open, partially closed, and fully closed eyes
      if (ear < EAR_THRESHOLD_CLOSED) {
        eyeClosedFrames++;
        eyePartiallyClosedFrames = 0;
        
        if (eyeClosedFrames >= CONSECUTIVE_FRAMES) {
          eyeStatus = 'CLOSED';
          updateAlert('🚨 EYES CLOSED', true);
        }
      } else if (ear < EAR_THRESHOLD_PARTIALLY) {
        eyePartiallyClosedFrames++;
        eyeClosedFrames = 0;
        
        if (eyePartiallyClosedFrames >= CONSECUTIVE_FRAMES) {
          eyeStatus = 'PARTIALLY_CLOSED';
          updateAlert('⚠️ EYES PARTIALLY CLOSED', true);
        }
      } else {
        eyeClosedFrames = 0;
        eyePartiallyClosedFrames = 0;
        eyeStatus = 'OPEN';
        updateAlert('🚨 EYES CLOSED', false);
        updateAlert('⚠️ EYES PARTIALLY CLOSED', false);
      }

      // ==================== YAWNING DETECTION ====================
      // Detect yawning based on Mouth Aspect Ratio
      if (mar > MAR_THRESHOLD_YAWN) {
        yawnFrames++;
        
        if (yawnFrames >= CONSECUTIVE_FRAMES) {
          yawnStatus = 'YAWNING';
          
          if (mar > MAR_THRESHOLD_ALERT) {
            updateAlert('⚠️ EXCESSIVE YAWNING', true);
          } else {
            updateAlert('⚠️ YAWNING DETECTED', true);
          }
        }
      } else {
        yawnFrames = 0;
        yawnStatus = 'NORMAL';
        updateAlert('⚠️ YAWNING DETECTED', false);
        updateAlert('⚠️ EXCESSIVE YAWNING', false);
      }

      // ==================== HEAD POSTURE DETECTION ====================
      // Detect head turns (yaw) using the universal ratio
      const isLookingLeft = headPose.yawRatio < YAW_RATIO_LEFT;
      const isLookingRight = headPose.yawRatio > YAW_RATIO_RIGHT;
      const headTurnAlert = isLookingLeft || isLookingRight;
      
      if (headTurnAlert) {
        headTurnedFrames++;
        
        if (headTurnedFrames >= CONSECUTIVE_FRAMES) {
          headStatus = 'TURNED';
          updateAlert('⚠️ HEAD NOT FORWARD', true);
        }
      } else {
        headTurnedFrames = 0;
        headStatus = 'FORWARD';
        updateAlert('⚠️ HEAD NOT FORWARD', false);
      }

      // ==================== OVERALL DRIVER STATUS ====================
      // Determine driver status based on active alerts
      if (currentAlerts.size > 0) {
        driverStatus = 'DROWSY';  // Emergency/Alert state
      } else {
        driverStatus = 'SAFE';    // Safe/Normal state
      }

      updateUI(ear, mar, headPose, eyeStatus, yawnStatus, headStatus, driverStatus);
      drawFaceOverlay(landmarks);
    }
  }

  updateFPS();
  animationFrameId = requestAnimationFrame(predictWebcam);
}

async function initializeFaceLandmarker() {
  try {
    // Ensure MediaPipe is initialized
    if (!FaceLandmarker || !FilesetResolver) {
      await initializeMediaPipe();
    }

    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.8/wasm"
    );

    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numFaces: 1,
      minFaceDetectionConfidence: 0.5,
      minFacePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
  } catch (error) {
    console.error('Error initializing FaceLandmarker:', error);
    throw error;
  }
}

async function startMonitoring() {
  try {
    if (!faceLandmarker) {
      startBtn.disabled = true;
      startBtn.innerHTML = '<span>Loading AI Model...</span>';
      await initializeFaceLandmarker();
    }

    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'user'
      }
    });

    video.srcObject = webcamStream;

    video.addEventListener('loadeddata', () => {
      videoPlaceholder.classList.add('hidden');
      predictWebcam();
    });

    startBtn.disabled = true;
    startBtn.classList.add('opacity-50', 'cursor-not-allowed');
    stopBtn.disabled = false;
    stopBtn.classList.remove('opacity-50', 'cursor-not-allowed');

    document.querySelector('header .text-gray-400').textContent = 'ACTIVE';
    document.querySelector('header .bg-red-500').classList.remove('bg-red-500');
    document.querySelector('header .bg-red-500, header .animate-pulse').classList.add('bg-green-500');

  } catch (error) {
    console.error('Error accessing camera:', error);
    alert('Failed to access camera. Please ensure camera permissions are granted.');
    startBtn.disabled = false;
    startBtn.innerHTML = '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg><span>Start Engine</span>';
  }
}

function stopMonitoring() {
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }

  video.srcObject = null;
  canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

  if (isAlertPlaying) {
    alertSound.pause();
    alertSound.currentTime = 0;
    isAlertPlaying = false;
  }
  currentAlerts.clear();
  alertPanel.classList.add('hidden');

  eyeClosedFrames = 0;
  eyePartiallyClosedFrames = 0;
  yawnFrames = 0;
  headTurnedFrames = 0;

  videoPlaceholder.classList.remove('hidden');

  startBtn.disabled = false;
  startBtn.classList.remove('opacity-50', 'cursor-not-allowed');
  stopBtn.disabled = true;
  stopBtn.classList.add('opacity-50', 'cursor-not-allowed');

  document.getElementById('driver-status').textContent = 'STANDBY';
  document.getElementById('driver-status').className = 'text-2xl font-bold text-gray-400';
  document.getElementById('status-indicator').className = 'w-2 h-2 rounded-full bg-gray-500';
  document.getElementById('eye-status').textContent = 'Unknown';
  document.getElementById('yawn-status').textContent = 'Unknown';
  document.getElementById('head-status').textContent = 'Unknown';
  document.getElementById('ear-value').textContent = 'EAR: --';
  document.getElementById('mar-value').textContent = 'MAR: --';
  document.getElementById('head-angle').textContent = 'Angle: --';
  document.getElementById('fps-counter').textContent = 'FPS: 0';

  document.querySelector('header .text-gray-400').textContent = 'STANDBY';
  const indicator = document.querySelector('header .animate-pulse');
  indicator.classList.remove('bg-green-500');
  indicator.classList.add('bg-red-500');
}

startBtn.addEventListener('click', startMonitoring);
stopBtn.addEventListener('click', stopMonitoring);