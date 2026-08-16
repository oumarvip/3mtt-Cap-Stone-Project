const imageInput = document.getElementById('imageInput');
const cropSelect = document.getElementById('crop');
const previewWrap = document.getElementById('previewWrap');
const previewImage = document.getElementById('previewImage');
const uploadText = document.getElementById('uploadText');
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
  uploadText.textContent = file.name;
}

imageInput.addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (file) {
    readImage(file);
  }
});

resetBtn.addEventListener('click', () => {
  selectedFile = null;
  imageInput.value = '';
  previewWrap.hidden = true;
  previewImage.src = '';
  uploadText.textContent = 'Choose an image or drag it here';
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
  }
});
