var formViewModel = Vue.createApp({
    data() {
        return {
            steelType: "粘度",
            castingWidth: "面积",
            billetWidth: "水口面积",
            billetThickness: "流速",
            mode: "模式",
            targetLV: "目标液位",
            predictButton: "预测"
        }
    }
}).mount('#steelInfoForm')

var seedCounter = 0;
function submitCastingPriorInfo(form, title = "预测结果") {
    axios.get(baseServerAddr + '/getcastingcurvefrominput', {
        Headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            Accept: "application/json"
        }
    }).then(response => {
        if (chartCollectionVueModel.isEmpty) {
            chartCollectionVueModel.chartViewModels.splice(0, 1);
            chartCollectionVueModel.isEmpty = false;
        }
        vmId = hashCode(title, seedCounter++);
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


function onTabClicked() {
    console.log("ontabclicked")
    chartCollectionVueModel.selectItem(chartCollectionVueModel.currentId)
}

function onUpdateClicked() {
    console.log("Updating model");
}


var chartCollectionVueModel = Vue.createApp({
    data() {
        return {
            isEmpty: true,
            chartViewModels: [
                ChartViewModel()
            ],
            currentId: 0
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
        }
    }
}).mount("#chartSection");

function ChartViewModel() {
    var defaultChartViewModel = {
        chartId: 0,
        title: "没数据",
        data: null,
        isPredictResult: false,
        isSelected: true,
        echart: null,
    };

    return defaultChartViewModel;
}



function showCastingCurve(data, title, id) {
    var newDataViewModel = ChartViewModel();
    newDataViewModel.chartId = id,
        newDataViewModel.title = title,
        newDataViewModel.data = data.castingCurveValues,
        newDataViewModel.isPredictResult = false,
        newDataViewModel.isSelected = true

    chartCollectionVueModel.addAndDisplayChart(newDataViewModel);
}

function exportToCsv() {
    var viewModel = null;
    for (var i = 0; i < chartCollectionVueModel.chartViewModels.length; ++i) {
        if (chartCollectionVueModel.currentId == chartCollectionVueModel.chartViewModels[i].chartId) {
            viewModel = chartCollectionVueModel.chartViewModels[i];
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
    var echart = echarts.init(document.getElementById(divId));
    currentChart = echart
    dataViewModel.echart = echart;
    var option = {
        xAxis: {
            type: 'category',
            data: dataViewModel.data.times
        },
        yAxis: [
            {
                type: 'value',
                min: -2
            },
            {
                type: 'value',
                min: -2
            }

        ],
        series: [
            {
                data: dataViewModel.data.values,
                yAxis: 0,
                type: 'line'
            },
            {
                data: dataViewModel.data.liqLevel,
                yAxis: 2,
                type: 'line'
            }
        ],
    };

    echart.setOption(option)

    //_animationStarted = true;
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
        if (chartCollectionVueModel.isEmpty) {
            chartCollectionVueModel.chartViewModels.splice(0, 1);
            chartCollectionVueModel.isEmpty = false;
        }
        const vmId = hashCode(fileName, seedCounter++);

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
    chartCollectionVueModel.revmoveItem(idToDelete)

}

function resizeEverything() {

    for (var i = 0; i < chartCollectionVueModel.chartViewModels.length; ++i) {
        if (chartCollectionVueModel.chartViewModels[i].chartId == chartCollectionVueModel.currentId) {
            if (chartCollectionVueModel.chartViewModels[i].echart != null) {
                chartCollectionVueModel.chartViewModels[i].echart.resize();
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
    chartCollectionVueModel.chartViewModels.forEach(element => {
        if (element.chartId == id) {
            requestViewModel = element;
        }
    });

    return requestViewModel;
}

function onPlayButtonClicked() {
    currentViewModel = getChartViewModelById(chartCollectionVueModel.currentId);
    if (currentViewModel.data != null) {
        playAnimation(currentViewModel.data);
    }
}

function playAnimation(data) {
    translateData = data;
    startTime = Date.now();
}