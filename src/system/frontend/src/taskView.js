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
        }
    }
}).mount("#taskListTable");

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
            axios.get(baseServerAddr + '/createTrainningTask?id=' + id, {
                Headers: {
                    "Content-Type": "application/x-www-form-urlencoded",
                    Accept: "application/json"
                }
            }).then(response => {
                taskCollectionVueModel.taskViewModels.push(response.data);
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