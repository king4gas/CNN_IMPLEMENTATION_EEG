const fileInput = document.getElementById("mat-file");
const fileNameDisplay = document.getElementById("filename");

fileInput.addEventListener("change", function () {
    fileNameDisplay.textContent = this.files[0]?.name || "No file selected";
});

function startUpload() {
    const file = fileInput.files[0];
    if (!file) return alert("Please select a .mat file.");

    const formData = new FormData();
    formData.append("file", file);

    document.getElementById("progress-container").style.display = "block";

    fetch("/train", { method: "POST", body: formData }).then(response => {
        if (!response.ok) alert("Upload or training failed.");
    });

    const progressBar = document.getElementById("progress-bar");
    const progressText = document.getElementById("progress-text");
    let lastProgress = 0;

    const interval = setInterval(() => {
        fetch('/progress')
            .then(response => response.json())
            .then(data => {
                if (data.progress >= lastProgress) {
                    progressBar.value = data.progress;
                    progressText.textContent = data.progress + '%';
                    lastProgress = data.progress;
                }

                if (data.progress >= 100) {
                    clearInterval(interval);
                    progressBar.value = 100;
                    progressText.textContent = '100%';
                    document.getElementById("view-result-btn").style.display = "inline-block";
                }
            });
    }, 1000);
}

function goToResult() {
    window.location.href = "/train_result";
}