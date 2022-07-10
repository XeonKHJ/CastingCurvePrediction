// TODO add gesture and trackpad support.
// ref: https://kenneth.io/post/detecting-multi-touch-trackpad-gestures-in-javascript

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
            isLvActVisible: true,
            isTudishWeightVisible: true,
            isLadleWeightVisible: true
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
    var chart = echarts.init(document.getElementById(divId));
    chart.vmId = dataViewModel.chartId
    chartManager.charts.push(chart)
    displayOrRefreshCastingChart(dataViewModel)
    //_animationStarted = true;
}

function displayOrRefreshCastingChart(dataViewModel) {
    echart = chartManager.getChartByChartId(dataViewModel.chartId)

    yAxises = []
    ySeries = []
    if (chartCollectionViewModel.isStpPosVisible) {
        yAxises.push({
            name: 'mm',
            type: 'value',
            min: -2,
            gridIndex: 0
        })
        ySeries.push({
            name: '塞棒位置',
            data: dataViewModel.data.stpPos,
            type: 'line',
            gridIndex: 0,
            xAxisIndex: 0,
            yAxisIndex: 0
        })
    }

    if (chartCollectionViewModel.isLvActVisible) {

        ySeries.push({
            name: '液位',
            data: dataViewModel.data.liqLevel,
            type: 'line',
            gridIndex: 0,
            xAxisIndex: 0,
            yAxisIndex: 0
        })
    }

    
    if (chartCollectionViewModel.isLadleWeightVisible) {
        yAxises.push({
            type: 'value',
            min: 0,
            gridIndex: 1
        });
        ySeries.push({
            name: '大包重量',
            data: dataViewModel.data.ladleWeights,
            type: 'line',
            gridIndex: 1,
            xAxisIndex: 1,
            yAxisIndex: 1
        })
    }

    if (chartCollectionViewModel.isTudishWeightVisible) {
        yAxises.push({
            name: 'mm',
            type: 'value',
            min: 0,
            gridIndex: 2
        });
        ySeries.push({
            name: '中包重量',
            data: dataViewModel.data.tudishWeights,
            type: 'line',
            gridIndex: 2,
            xAxisIndex: 2,
            yAxisIndex: 2
        })
    }



    if (ySeries.length == 0) {
        yAxises.push({
            type: 'value',
            min: -2
        })
        ySeries.push({
            data: [],
            yAxis: 0,
            type: 'line'
        })
    }

    const data = [["2000-06-05", 116], ["2000-06-06", 129], ["2000-06-07", 135], ["2000-06-08", 86], ["2000-06-09", 73], ["2000-06-10", 85], ["2000-06-11", 73], ["2000-06-12", 68], ["2000-06-13", 92], ["2000-06-14", 130], ["2000-06-15", 245], ["2000-06-16", 139], ["2000-06-17", 115], ["2000-06-18", 111], ["2000-06-19", 309], ["2000-06-20", 206], ["2000-06-21", 137], ["2000-06-22", 128], ["2000-06-23", 85], ["2000-06-24", 94], ["2000-06-25", 71], ["2000-06-26", 106], ["2000-06-27", 84], ["2000-06-28", 93], ["2000-06-29", 85], ["2000-06-30", 73], ["2000-07-01", 83], ["2000-07-02", 125], ["2000-07-03", 107], ["2000-07-04", 82], ["2000-07-05", 44], ["2000-07-06", 72], ["2000-07-07", 106], ["2000-07-08", 107], ["2000-07-09", 66], ["2000-07-10", 91], ["2000-07-11", 92], ["2000-07-12", 113], ["2000-07-13", 107], ["2000-07-14", 131], ["2000-07-15", 111], ["2000-07-16", 64], ["2000-07-17", 69], ["2000-07-18", 88], ["2000-07-19", 77], ["2000-07-20", 83], ["2000-07-21", 111], ["2000-07-22", 57], ["2000-07-23", 55], ["2000-07-24", 60]];
    const dateList = data.map(function (item) {
        return item[0];
    });
    const valueList = data.map(function (item) {
        return item[1];
    });
    option = {
        title: [
            {
                left: 'center',
                text: '塞棒位置'
            },
            {
                top: '35%',
                left: 'center',
                text: '中包重量'
            },
            {
                top: '70%',
                left: 'center',
                text: '大包重量'
            }
        ],
        tooltip: {
            trigger: 'axis',
            axisPointer: {
                animation: false
            }
        },
        xAxis: [
            {
                data: dataViewModel.data.times,
                gridIndex: 0
            },
            {
                data: dataViewModel.data.times,
                gridIndex: 1
            },
            {
                data: dataViewModel.data.times,
                gridIndex: 2
            }
        ],
        yAxis: yAxises,
        legend: {
            data: ['塞棒位置', '液位', '中包重量','大包重量'],
            left: 10
        },
        grid: [
            {
                bottom: '70%'
            },
            {
                top: '32.5%',
                bottom: '37.5%'
            },
            {
                top: '65%',
                bottom: '5%'
            }
        ],
        series: ySeries,
        axisPointer: {
            link: [
                {
                    xAxisIndex: 'all'
                }
            ]
        },
        dataZoom: [
            {
                show: true,
                realtime: true,
                xAxisIndex: [0, 1, 2]
            },
            {
                type: 'inside',
                realtime: true,
                xAxisIndex: [0, 1, 2]
            }
        ],
        toolbox: {
            show: true,
            feature: {
                myTool1: {
                    show: true,//是否显示    
                    title: '导出', //鼠标移动上去显示的文字    
                    icon: 'path://M525.4 721.2H330.9c-9 0-18.5-7.7-18.5-18.1V311c0-9 9.3-18.1 18.5-18.1h336.6c9.3 0 18.5 9.1 18.5 18.1v232.7c0 6 8.8 12.1 15 12.1 6.2 0 15-6 15-12.1V311c0-25.6-25.3-48.9-50.1-48.9h-335c-26.2 0-50.1 23.3-50.1 48.9v389.1c0 36.3 20 51.5 50.1 51.5h197.6c6.2 0 9.3-7.5 9.3-15.1 0-6-6.2-15.3-12.4-15.3zM378.8 580.6c-6.2 0-12.3 8.6-12.3 14.6s6.2 14.6 12.3 14.6h141.4c6.2 0 12.3-5.8 12.3-13.4 0.3-9.5-6.2-15.9-12.3-15.9H378.8z m251.6-91.2c0-6-6.2-14.6-12.3-14.6H375.7c-6.2 0-12.4 8.6-12.4 14.6s6.2 14.6 12.4 14.6h240.8c6.2 0.1 13.9-8.5 13.9-14.6z m-9.2-120.5H378.8c-6.2 0-12.3 8.6-12.3 14.6s6.2 14.6 12.3 14.6h240.8c7.7 0 13.9-8.6 13.9-14.6s-6.2-14.6-12.3-14.6z m119.4 376.6L709 714.1c9.2-12 14.6-27 14.6-43.2 0-39.4-32.1-71.4-71.8-71.4-39.7 0-71.8 32-71.8 71.4s32.1 71.4 71.8 71.4c16.3 0 31.3-5.4 43.4-14.5l31.6 31.5c3.8 3.8 10 3.8 13.8 0 3.8-3.8 3.8-10 0-13.8z m-88.8-23.6c-28.3 0-51.3-22.8-51.3-51s23-51 51.3-51c28.3 0 51.3 22.8 51.3 51s-23 51-51.3 51z', //图标    
                    onclick: function () {//点击事件,这里的option1是chart的option信息    
                        exportToCsv();
                    }
                },
                mark: { show: true },
                dataView: {
                    show: true,
                    readOnly: false,
                    //修改数据视图格式
                    optionToContent: function (opt) {
                        console.log("fskdjflsd")
                    }
                },
                magicType: { show: true, type: ['line', 'bar'] },
                restore: { show: true },
                saveAsImage: { show: true },
                dataZoom: {
                    show: true,
                },
            }
        }
    };
    echart.setOption(option, true)
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

    drawAnimation()

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

const aniSectionAndChartSectionSplitter = {
    init() {
        let aniSectionAndChartSectionSplitBarDiv = document.getElementById("aniSectionAndChartSectionSplitBarDiv");
        aniSectionAndChartSectionSplitBarDiv.onmousedown = (param) => {
            console.log("down")
            console.log(param)
            // aniSectionAndChartSectionSplitBarDiv.onmousemove = (param) =>{
            //     console.log("mouse move")
            //     console.log(param)
            // }
            document.addEventListener('mouseup', mouseUpFromMovingAniSectionAndChartSectionSplitBarDiv);
            document.addEventListener('mousemove', moveAniSectionAndChartSectionSplitBarDiv);
        }


        aniSectionAndChartSectionSplitBarDiv.ondrag = (e) => {


        }

        aniSectionAndChartSectionSplitBarDiv.onmouseup = (param) => {
            // Remove all event listener here.

            console.log("up")
            console.log(param)
            // aniSectionAndChartSectionSplitBarDiv.onmousemove = null;
        }
    }
}

function mouseUpFromMovingAniSectionAndChartSectionSplitBarDiv() {
    console.log("mouseUp")
    
    document.removeEventListener("mousemove", moveAniSectionAndChartSectionSplitBarDiv)
    document.removeEventListener("mouseup", mouseUpFromMovingAniSectionAndChartSectionSplitBarDiv)
    resizeEverything()
}

function moveAniSectionAndChartSectionSplitBarDiv(e)
{
    console.log("moveDoms")
    const percentage = e.clientX / document.documentElement.scrollWidth;
    if (percentage == 0) {
        resizeEverything()
        return;
    }
    aniSectionAndChartSectionSplitBarDiv.style['right'] = (1 - percentage) * 100 + '%'
    console.log(aniSectionAndChartSectionSplitBarDiv.style['right'])

    const chartSection = document.getElementById("chartSection");
    chartSection.style['width'] = (1 - percentage) * 100 + '%'
    const animationSection = document.getElementById('animationSection');
    animationSection.style['width'] = percentage * 100 + '%'
    resizeEverything()
}

aniSectionAndChartSectionSplitter.init()
