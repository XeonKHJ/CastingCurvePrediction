function TaskViewModel(id = 0, loss = 0.0, status = "Stopped", modelId) {
    return {
        id: id,
        loss: loss,
        status: status,
        modelId: modelId
    }
}

function ModelViewModel(id = 0, loss = 0.0, status = "untrained") {
    return {
        id: 0,
        loss: 0.0,
        path: "don't know",
        status: status
    }
}

var taskCollectionVueModel = Vue.createApp({
    data() {
        return {
            isEmpty: true,
            taskViewModels: [
                TaskViewModel()
            ],
            currentId: 0
        }
    },
    methods: {
        onDeleteButtonClicked(id, idx) {
            axios.get(baseServerAddr + '/deleteTaskById?id=' + id, {
                Headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                    Accept: "application/json"
                }
            }).then(response => {
                taskCollectionVueModel.taskViewModels.splice(idx, 1);
            }).then().catch(
                err => {
                    console.log(err)
                    showError(err)
                })
        },
        onStartButtonClicked(task) {
            task.status = "Starting"
            axios.get(baseServerAddr + '/startTrainingTask?taskId=' + task.id, {
                Headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                    Accept: "application/json"
                }
            }).then(response => {
                switch (response.data.statusCode) {
                    case 1:
                        task.status = "Running";
                        break;
                    default:
                        task.status = "Stopped"
                        showError(response.message);
                }
            }).then().catch(
                err => {
                    task.status = "Stopped"
                    console.log(err)
                    showError(err)
                })
        },
        // TODO Stop task.
        onStopButtonClicked(task, idx) {
            axios.get(baseServerAddr + '/stopTaskById?taskId=' + task.id, {
                Headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                    Accept: "application/json"
                }
            }).then(response => {
                taskCollectionVueModel.taskViewModels.splice(idx, 1);
            }).then().catch(
                err => {
                    console.log(err)
                    showError(err)
                })
        },
        onViewTaskButtonClicked(task) {
            window.location.href = ("./task.html?taskId=" + task.id);
        }
    }
}).mount("#taskListTable");

function getStatusByTaskId(taskId) {
    axios.get(baseServerAddr + '/getTaskStatus?taskId=' + taskId, {
        Headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            Accept: "application/json"
        }
    }).then(response => {
        switch (response.statusCode) {
            case 1:
                task.status = "Training";
            default:
                task.status = "Stopped"
                showError(response.message);
        }
    }).then().catch(
        err => {
            task.status = "Stopped"
            console.log(err)
            showError(err)
        })
}


var modelCollectionVueModel = Vue.createApp({
    data() {
        return {
            isEmpty: true,
            modelViewModels: [
                ModelViewModel()
            ],
            currentId: 0
        }
    },
    methods: {
        onCreateTaskButtonClicked(id) {
            axios.get(baseServerAddr + '/createTrainingTaskFromModelId?modelId=' + id, {
                Headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                    Accept: "application/json"
                }
            }).then(response => {
                if(response.data.statusCode <= 0)
                {
                    showError(response.data.message)
                }
                else
                {
                    taskCollectionVueModel.taskViewModels.push(response.data);
                }
            }).then().catch(
                err => {
                    console.log(err)
                    showError(err)
                }
            )
        },
        onDeleteButtonClicked(id, idx) {
            axios.get(baseServerAddr + '/deleteModelById?id=' + id, {
                Headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                    Accept: "application/json"
                }
            }).then(response => {
                if (response.data.statusCode < 0) {
                    showError(response.data.message)
                }
                else {
                    modelCollectionVueModel.modelViewModels.splice(idx, 1);
                }

            }).then().catch(
                err => {
                    console.log(err)
                    showError(err)
                })
        }
    }
}).mount("#modelListTable");

function getTasks() {
    axios.get(baseServerAddr + '/getTasks', {
        Headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            Accept: "application/json"
        }
    }).then(response => {
        for (i = 0; i < response.data.taskViewModels.length; ++i) {
            taskCollectionVueModel.taskViewModels.push(response.data.taskViewModels[i]);
        }
    }).then().catch(
        err => {
            console.log(err)
        })
}

function getModels() {
    axios.get(baseServerAddr + '/getModels', {
        Headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            Accept: "application/json"
        }
    }).then(response => {
        for (i = 0; i < response.data.mlModelViewModels.length; ++i) {
            modelCollectionVueModel.modelViewModels.push(response.data.mlModelViewModels[i]);
        }
    }).then().catch(
        err => {
            console.log(err)
        })
}

function onCreateModelButtonClicked() {
    axios.get(baseServerAddr + '/createModel', {
        Headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            Accept: "application/json"
        }
    }).then(response => {
        modelCollectionVueModel.modelViewModels.push(response.data);
    }).then().catch(
        err => {
            console.log(err)
            showError(err)
        })
}

getModels();
getTasks();