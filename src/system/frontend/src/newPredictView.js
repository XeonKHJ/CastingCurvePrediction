// Cross js file variables
var context = {
    translateData: null,
    animationStartTime: null
}

// ViewModels section
var formViewModel = Vue.createApp({
    data() {
        return {
            steelType: "粘度",
            castingWidth: "面积",
            billetWidth: "水口面积",
            billetThickness: "流速",
            mode: "模式",
            targetLV: "目标液位",
            predictButton: "预测",
            isEmpty: true,
            steelGrades: [
                "SPHC(C0.06)",
                "Q235B",
                "SPA-H"
            ],
            modelViewModels: [

            ],
            currentId: 0
        }
    }
}).mount('#steelInfoForm')

// ViewModel Data Id Generator
const dataIdGenerator = {
    seedCounter: 0,
    generateId(title) {
        vmId = hashCode(title, this.seedCounter++);
        return vmId
    }
}

const chartManager = {
    charts: [],
    getChartByChartId(id) {
        for (var i = 0; i < this.charts.length; ++i) {
            if (this.charts[i].vmId == id) {
                return this.charts[i]
            }
        }
        return null;
    },
    deleteChartById(id) {
        for (var i = 0; i < this.charts.length; ++i) {
            if (this.charts[i].chartId == id) {
                this.charts.splice(i, 1);
                break;
            }
        }
    }
}

function submitCastingPriorInfo(form, title = "预测结果") {
    axios.get(baseServerAddr + '/getcastingcurvefrominput', {
        Headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            Accept: "application/json"
        }
    }).then(response => {
        if (chartCollectionViewModel.isEmpty) {
            chartCollectionViewModel.chartViewModels.splice(0, 1);
            chartCollectionViewModel.isEmpty = false;
        }
        vmId = dataIdGenerator.generateId(title);
        var filePaths = title.split('\\');
        var fileName = filePaths[filePaths.length - 1]
        showCastingCurve(response.data, fileName, vmId);
    }).then().catch(
        err => {
            console.log(err)
            showError(err)
        }
    )
}

var chartListViewModel = Vue.createApp({
    data() {
        return {
            tabTitles: ['没有数据'],
            tabDatas: [],
            currentIndex: 0,
        }
    }
})


var chartCollectionViewModel = Vue.createApp({
    data() {
        return {
            isEmpty: true,
            chartViewModels: [
                ChartViewModel()
            ],
            currentId: 0,
            isStpPosVisible: true,
            isLvActVisible: false,
            isTudishWeightVisible: false,
            isLadleWeightVisible: false
        }
    },
    methods: {
        revmoveItem(idToDelete) {
            var length = this.chartViewModels.length;
            var echartToDispose;

            for (i = 0; i < length; ++i) {
                if (this.chartViewModels[i].chartId == idToDelete) {
                    var element = this.chartViewModels[i];
                    echartToDispose = element;

                    if (length == 1) {
                        this.chartViewModels.push(ChartViewModel());
                        this.selectItem(0);
                        this.isEmpty = true;
                    }
                    else if (this.currentId == idToDelete) {

                        if (length == i + 1) {
                            this.selectItem(this.chartViewModels[i - 1].chartId);
                        }
                        else {
                            var chartId = this.chartViewModels[i + 1].chartId;
                            var el = document.getElementById('echartContent' + chartId);
                            this.selectItem(this.chartViewModels[i + 1].chartId);
                        }
                    }

                    if (echartToDispose.echart != null) {
                        echartToDispose.echart.clear();
                        echartToDispose.echart.dispose();
                    }

                    this.chartViewModels.splice(i, 1);
                    break;
                }
            }
        },
        getItemFromId(id) {
            if (id == 0) {
                return null
            }
            else {
                var result = true;
                this.chartViewModels.forEach(element => {
                    if (element.chartId == id) {
                        result = element;
                    }
                });

                return result
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

            chartCollectionViewModel.$nextTick(() => {
                resizeEverything();
            });
        },
        addAndDisplayChart(newDataViewModel) {
            const id = newDataViewModel.chartId;
            if (this.isEmpty) {
                this.chartViewModels.splice(0, 1);
                this.isEmpty = false;
            }

            this.chartViewModels.push(newDataViewModel);
            this.$nextTick(() => {
                generatingCastingChart(newDataViewModel);
                this.selectItem(id);
            });
        },
        onChartDataContentChanged() {
            refreshCastingChart();
        },
        onPlayButtonClicked() {
            curItem = this.getItemFromId(this.currentId)

            if (curItem == null || curItem == undefined) {
                showError("得先有条线吧？")
            }
            else {
                playAnimation(curItem.data);
            }

        }
    }
}).mount("#chartSection");

function onStpPosChecked() {
    console.log("sdfsdf")
}

function ChartViewModel() {
    var defaultChartViewModel = {
        chartId: 0,
        title: "没数据",
        data: null,
        isPredictResult: false,
        isSelected: true,
        isStpPosVisible: true,
    };

    return defaultChartViewModel;
}

// other functions section;
function onTabClicked() {
    console.log("ontabclicked")
    chartCollectionViewModel.selectItem(chartCollectionViewModel.currentId)
}

function onUpdateClicked() {
    console.log("Updating model");
}


function showCastingCurve(data, title, id) {
    var newDataViewModel = ChartViewModel();
    newDataViewModel.chartId = id;
    newDataViewModel.title = title;
    newDataViewModel.data = data.castingCurveValues;
    newDataViewModel.isPredictResult = false;
    newDataViewModel.isSelected = true;

    chartCollectionViewModel.addAndDisplayChart(newDataViewModel);
}

function exportToCsv() {
    var viewModel = null;
    for (var i = 0; i < chartCollectionViewModel.chartViewModels.length; ++i) {
        if (chartCollectionViewModel.currentId == chartCollectionViewModel.chartViewModels[i].chartId) {
            viewModel = chartCollectionViewModel.chartViewModels[i];
            break;
        }
    }

    save(viewModel.title, castingCurveDataToCsv(viewModel.data));
}

function castingCurveDataToCsv(data) {
    var str = "times, values\n";
    for (var i = 0; i < data.times.length; ++i) {
        str += String(data.times[i]) + ", " + String(data.values[i]) + "\n";
    }
    return str;
}

function save(filename, data) {
    const blob = new Blob([data], { type: 'text/csv' });
    if (window.navigator.msSaveOrOpenBlob) {
        window.navigator.msSaveBlob(blob, filename);
    }
    else {
        const elem = window.document.createElement('a');
        elem.href = window.URL.createObjectURL(blob);
        elem.download = filename;
        document.body.appendChild(elem);
        elem.click();
        document.body.removeChild(elem);
    }
}

var currentChart = null;
function generatingCastingChart(dataViewModel) {
    var divId = createEchartContentDiv(dataViewModel.title, dataViewModel.chartId)
    // var chart = echarts.init(document.getElementById(divId));
    // chart.vmId = dataViewModel.chartId
    // chartManager.charts.push(chart)
    displayOrRefreshCastingChart(dataViewModel)
}

function displayOrRefreshCastingChart(dataViewModel) {
    const myChart1 = new Chart(document.getElementById('chartCanvas').getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
            datasets: [{
                label: '# of Votes',
                data: [12, 19, 3, 5, 2, 3],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 159, 64, 0.2)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                },

            },
            maintainAspectRatio : false,
            responsive: true
        }
    });
    const myChart2 = new Chart(document.getElementById('chartCanvas2').getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
            datasets: [{
                label: '# of Votes',
                data: [12, 19, 3, 5, 2, 3],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 159, 64, 0.2)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                },

            },
            maintainAspectRatio : false,
            responsive: true
        }
    });
    const myChart3 = new Chart(document.getElementById('chartCanvas3').getContext('2d'), {
        type: 'bar',
        data: {
            labels: ['Red', 'Blue', 'Yellow', 'Green', 'Purple', 'Orange'],
            datasets: [{
                label: '# of Votes',
                data: [12, 19, 3, 5, 2, 3],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(75, 192, 192, 0.2)',
                    'rgba(153, 102, 255, 0.2)',
                    'rgba(255, 159, 64, 0.2)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            },
            maintainAspectRatio : false,
            responsive: true
        }
    });
}

function refreshCastingChart() {
    item = chartCollectionViewModel.getItemFromId(chartCollectionViewModel.currentId)
    displayOrRefreshCastingChart(item)
}

function createEchartContentDiv(name, chartId) {
    var divId = "echartContent" + chartId;
    return divId;
}


function onOpenFileButtonClicked() {
    var openFileButton = document.getElementById('openFileInput');
    openFileButton.click();
}

function onOpenFileInputChanged(event) {
    var openFileInput = document.getElementById('openFileInput');
    var filePath = openFileInput.value;

    var filePaths = filePath.split('\\');
    var fileName = filePaths[filePaths.length - 1]
    const formData = new FormData();
    formData.append("file", event.files[0]);
    axios.post(baseServerAddr + '/openCastingCurveFile', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
            Accept: "application/json"
        }
    }).then(response => {
        openFileInput.value = "";
        if (chartCollectionViewModel.isEmpty) {
            chartCollectionViewModel.chartViewModels.splice(0, 1);
            chartCollectionViewModel.isEmpty = false;
        }
        const vmId = dataIdGenerator.generateId(filename);

        showCastingCurve(response.data, fileName, vmId);
    }).then().catch(
        err => console.log(err))
    openFileInput.value = '';
}

window.onresize = function () {
    resizeEverything();
}

function onCloseTabButtonClicked(el) {

    idToDelete = el.id.substring(8, el.id.length)
    chartCollectionViewModel.revmoveItem(idToDelete)

}

function resizeEverything() {

    for (var i = 0; i < chartManager.charts.length; ++i) {
        if (chartManager.charts[i].vmId == chartCollectionViewModel.currentId) {
            if (chartManager.charts[i] != null) {
                chartManager.charts[i].resize();
            }
            break;
        }
    }

    var canvas = document.getElementById('animationCanvas');
    var canvasDiv = document.getElementById('animationDiv');
    // canvas.Height = canvasDiv.Height;
    //startAnimation()

    console.log("(" + canvasDiv.clientHeight + "," + canvasDiv.clientWidth + ")")

    canvas.height = canvasDiv.clientHeight - 10;
    canvas.width = canvasDiv.clientWidth - 10;
}


function getChartViewModelById(id) {
    var requestViewModel = null;
    chartCollectionViewModel.chartViewModels.forEach(element => {
        if (element.chartId == id) {
            requestViewModel = element;
        }
    });

    return requestViewModel;
}

function playAnimation(data) {
    AnimationController.startAnimation(data)
}


function loadTrainedModels() {
    axios.get(baseServerAddr + '/getTrainedModels', {
        Headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            Accept: "application/json"
        }
    }).then(response => {
        for (i = 0; i < response.data.mlModelViewModels.length; ++i) {
            formViewModel.modelViewModels.push(response.data.mlModelViewModels[i]);
        }
    }).then().catch(
        err => {
            console.log(err)
        })
}

loadTrainedModels();