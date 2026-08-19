const imageInput = document.getElementById('imageInput');
const cameraInput = document.getElementById('cameraInput');
const cameraBtn = document.getElementById('cameraBtn');
const cameraDialog = document.getElementById('cameraDialog');
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const cameraError = document.getElementById('cameraError');
const captureBtn = document.getElementById('captureBtn');
const closeCameraBtn = document.getElementById('closeCameraBtn');
const cancelCameraBtn = document.getElementById('cancelCameraBtn');
const cropSelect = document.getElementById('crop');
const previewWrap = document.getElementById('previewWrap');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const statusBox = document.getElementById('statusBox');
const resultCard = document.getElementById('resultCard');
const labelBadge = document.getElementById('labelBadge');
const confidenceText = document.getElementById('confidenceText');
const confidenceBar = document.getElementById('confidenceBar');
const displayName = document.getElementById('displayName');
const recommendation = document.getElementById('recommendation');

let selectedFile = null;
let previewUrl = null;
let cameraStream = null;

function setStatus(message, type = 'idle') {
  statusBox.textContent = message;
  statusBox.className = `status-box ${type}`;
}

function readImage(file) {
  if (!file) return;

  selectedFile = file;
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewWrap.hidden = false;
  analyzeBtn.disabled = false;
  setStatus(`${file.name} is ready to analyze.`, 'idle');
}

function stopCamera() {
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraVideo.srcObject = null;
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    cameraInput.click();
    return;
  }

  cameraError.hidden = true;
  cameraDialog.showModal();
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' } },
      audio: false,
    });
    cameraVideo.srcObject = cameraStream;
  } catch (error) {
    cameraError.textContent = 'Camera access was not available. Choose a photo instead.';
    cameraError.hidden = false;
  }
}

function closeCamera() {
  stopCamera();
  if (cameraDialog.open) cameraDialog.close();
}

imageInput.addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (file) {
    readImage(file);
  }
});

cameraInput.addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (file) readImage(file);
});

cameraBtn.addEventListener('click', openCamera);
closeCameraBtn.addEventListener('click', closeCamera);
cancelCameraBtn.addEventListener('click', closeCamera);

captureBtn.addEventListener('click', () => {
  if (!cameraVideo.videoWidth) {
    cameraError.textContent = 'The camera is still starting. Try again in a moment.';
    cameraError.hidden = false;
    return;
  }

  cameraCanvas.width = cameraVideo.videoWidth;
  cameraCanvas.height = cameraVideo.videoHeight;
  cameraCanvas.getContext('2d').drawImage(cameraVideo, 0, 0);
  cameraCanvas.toBlob((blob) => {
    if (blob) readImage(new File([blob], `leaf-photo-${Date.now()}.jpg`, { type: 'image/jpeg' }));
    closeCamera();
  }, 'image/jpeg', 0.92);
});

resetBtn.addEventListener('click', () => {
  selectedFile = null;
  imageInput.value = '';
  cameraInput.value = '';
  previewWrap.hidden = true;
  previewImage.src = '';
  analyzeBtn.disabled = true;
  resultCard.classList.add('hidden');
  setStatus('Waiting for image upload...', 'idle');
});

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) {
    setStatus('Please select an image before analyzing.', 'error');
    return;
  }

  const formData = new FormData();
  formData.append('image', selectedFile);
  formData.append('crop', cropSelect.value);

  setStatus('Analyzing image...', 'loading');
  analyzeBtn.disabled = true;
  resultCard.classList.add('hidden');

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      throw new Error(data.error || 'Prediction failed.');
    }

    labelBadge.textContent = data.display_name;
    confidenceText.textContent = `${data.confidence}%`;
    displayName.textContent = data.display_name;
    recommendation.textContent = data.recommendation;
    confidenceBar.style.width = `${Math.min(data.confidence, 100)}%`;

    resultCard.classList.remove('hidden');
    setStatus('Prediction complete.', 'success');
  } catch (error) {
    setStatus(error.message || 'Something went wrong. Please try again.', 'error');
  } finally {
    analyzeBtn.disabled = false;
  }
});
