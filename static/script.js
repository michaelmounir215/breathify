const form = document.getElementById('uploadForm');
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(form);

    const resultDiv = document.getElementById('result');
    resultDiv.textContent = "Analyzing...";

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        resultDiv.textContent = `Prediction: ${result.top_disease} (${(result.probability * 100).toFixed(2)}%)`;
    } catch (error) {
        resultDiv.textContent = "Error analyzing image.";
        console.error(error);
    }
});
