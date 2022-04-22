var uploadInputViewModel = Vue.createApp({
    data() {
        return {
            datasetPath: "数据集路径",
            data: "",
            status: "无数据集",
            isTraining: false
        }
    }
}).mount('#uploadDatasetDiv')

var trainingStatusViewModel = Vue.createApp({
    data() {
        return
        {
            accuracy: "准确率"
        }
    }
})

var files = null;

function onUploadButtonClick() {
    var uploadDatasetInput = document.getElementById("uploadDatasetInput");
    uploadDatasetInput.click();
}

function onUploadDatasetInputChanged(event) {
    var uploadDatasetInput = document.getElementById("uploadDatasetInput");
    uploadInputViewModel.datasetPath = uploadDatasetInput.value;
    files = event.files;
}

var isUploading = true;
function getUploadStatus() {
    if (isUploading) {
        uploadInputViewModel.status = "正在上传……";
        uploadInputViewModel.isTraining = true;
    }
}

function onTrainButtonClicked() {
    const formData = new FormData();
    formData.append("file", files[0]);

    axios.post(baseServerAddr + '/uploadAndTrainModel', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
            Accept: "application/json"
        }
    }).then(response => {
        var data = response.data;
        uploadInputViewModel.status = data.status;
    })
    getUploadStatus();
}

function onStopTrainingClicked()
{
    console.log("正在停止训练");
}