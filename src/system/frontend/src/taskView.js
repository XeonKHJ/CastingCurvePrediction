function TaskViewModel(id = 0, loss = 0.0, status = "Stopped")
{
    return {
        id: 0,
        loss: 0.0,
        status: "Stopped"
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
        revmoveItem(idToDelete) {
            var length = this.taskViewModel.length;
            var echartToDispose;

            for (i = 0; i < length; ++i) {
                if (this.taskViewModel[i].chartId == idToDelete) {
                    var element = this.taskViewModel[i];
                    if (length == 1) {
                        this.taskViewModel.push(taskViewModel());
                        this.selectItem(0);
                        this.isEmpty = true;
                    }
                    else if (this.currentId == idToDelete) {

                        if (length == i + 1) {
                            this.selectItem(this.taskViewModels[i - 1].chartId);
                        }
                        else {
                            var chartId = this.taskViewModels[i + 1].chartId;
                            this.selectItem(this.taskViewModels[i + 1].chartId);
                        }
                    }
                    this.taskViewModel.splice(i, 1);
                    break;
                }
            }
        },
        selectItem(id) {
            if (id == 0) {
                var echartDiv = document.getElementById('echartDiv');
            }
            else {
                this.chartViewModels.forEach(element => {
                    if (element.chartId == id) {
                        element.isSelected = true;
                        this.currentId = element.chartId;
                    }
                    else {
                        element.isSelected = false;
                    }
                });
            }
            chartCollectionVueModel.$nextTick(() => {
            });
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
        for(i = 0; i < response.data.taskViewModels.length; ++i)
        {
            taskCollectionVueModel.taskViewModels.push(response.data.taskViewModels[i]);
        }
    }).then().catch(
        err => console.log(err))
}

getTasks();