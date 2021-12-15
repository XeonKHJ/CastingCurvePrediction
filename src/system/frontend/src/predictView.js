var formViewModel = Vue.createApp({
    data() {
        return {
            steelType: "粘度",
            castingWidth: "面积",
            billetWidth: "大包水口面积",
            billetThickness: "流速",
            mode: "铸机模式",
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
        if (chartCollectionViewModel.isEmpty) {
            chartCollectionViewModel.chartViewModels.splice(0, 1);
            chartCollectionViewModel.isEmpty = false;
        }
        vmId = hashCode(title, seedCounter++);
        var filePaths = title.split('\\');
        var fileName = filePaths[filePaths.length - 1]
        showCastingCurve(response.data, fileName, vmId);
    }).then().catch(
        err => console.log(err))
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
    console.log("Tab" + chartListViewModel.currentIndex + "is clicked");
    selectItem(chartCollectionViewModel.currentId)
}

function onUpdateClicked() {
    console.log("Updating model");
}


var defaultChartViewModel = {
    chartId: 0,
    title: "没数据",
    data: null,
    isPredictResult: false,
    isSelected: true
};

var chartCollectionViewModel = Vue.createApp({
    data() {
        return {
            isEmpty: true,
            chartViewModels: [
                {
                    chartId: 0,
                    title: "没数据",
                    data: null,
                    isPredictResult: false,
                    isSelected: true,
                    echart: null,
                }
            ],
            currentId: 0
        }
    }
}).mount("#chartSection");

function showCastingCurve(data, title, id) {
    var newDataViewModel = {
        chartId: vmId,
        title: title,
        data: data.castingCurveValues,
        isPredictResult: false,
        isSelected: true,
        echart: null
    }

    chartCollectionViewModel.chartViewModels.push(newDataViewModel);
    chartCollectionViewModel.$nextTick(() => {
        generatingCastingChart(newDataViewModel);
        selectItem(vmId);
        // startTime = Date.now();
        //_animationStarted = true;
    });
}

function exportToCsv() {
    var viewModel = null;
    for(var i = 0; i < chartCollectionViewModel.chartViewModels.length; ++i)
    {
        if(chartCollectionViewModel.currentId == chartCollectionViewModel.chartViewModels[i].chartId)
        {
            viewModel = chartCollectionViewModel.chartViewModels[i];
            break;
        }
    }

    save(viewModel.title, castingCurveDataToCsv(viewModel.data));
}

function castingCurveDataToCsv(data)
{
    var str = "times, values\n";
    for(var i = 0; i < data.times.length; ++i)
    {
        str+=String(data.times[i]) + ", "+ String(data.values[i]) + "\n";
    }
    return str;
}

function save(filename, data) {
    const blob = new Blob([data], {type: 'text/csv'});
    if(window.navigator.msSaveOrOpenBlob) {
        window.navigator.msSaveBlob(blob, filename);
    }
    else{
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
        yAxis: {
            type: 'value',
            min: -2
        },
        series: [
            {
                data: dataViewModel.data.values,
                type: 'line'
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

    echart.setOption(option)

    //_animationStarted = true;
}

function createEchartContentDiv(name, chartId) {
    var divId = "echartContent" + chartId;
    // Create tab page
    //var echartDiv = document.getElementById("echartDiv");
    //echartDiv.innerHTML = "<div class=\"echartContentDiv\" id=\""+divId+"\"></div>";
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
        if (chartCollectionViewModel.isEmpty) {
            chartCollectionViewModel.chartViewModels.splice(0, 1);
            chartCollectionViewModel.isEmpty = false;
        }
        vmId = hashCode(fileName, seedCounter++);

        showCastingCurve(response.data, fileName, vmId);
    }).then().catch(
        err => console.log(err))
}

window.onresize = function () {
    resizeEverything();
}

function onCloseTabButtonClicked() {
    var i = 0;
    var length = chartCollectionViewModel.chartViewModels.length;

    for (i = 0; i < length; ++i) {
        if (chartCollectionViewModel.chartViewModels[i].chartId == chartCollectionViewModel.currentId) {
            var element = chartCollectionViewModel.chartViewModels[i];

            if (element.echart != null) {
                element.echart.clear();
                element.echart.dispose();
            }

            chartCollectionViewModel.chartViewModels.splice(i, 1);
            break;
        }
    }

    if (length == 1) {
        chartCollectionViewModel.chartViewModels.push(defaultChartViewModel);
        chartCollectionViewModel.isEmpty = true;
    }
    else if (length == i + 1) {
        selectItem(chartCollectionViewModel.chartViewModels[i - 1].chartId);
    }
    else {
        selectItem(chartCollectionViewModel.chartViewModels[i].chartId);
    }
}

function resizeEverything() {

    for (var i = 0; i < chartCollectionViewModel.chartViewModels.length; ++i) {
        if (chartCollectionViewModel.chartViewModels[i].chartId == chartCollectionViewModel.currentId) {
            if (chartCollectionViewModel.chartViewModels[i].echart != null) {
                chartCollectionViewModel.chartViewModels[i].echart.resize();
            }
            break;
        }
    }

    var canvas = document.getElementById('animationCanvas');
    var canvasDiv = document.getElementById('animationDiv');
    // canvas.Height = canvasDiv.Height;
    //startAnimation()

    canvas.height = canvasDiv.clientHeight;
    canvas.width = canvasDiv.clientWidth;
}

function selectItem(id) {
    chartCollectionViewModel.chartViewModels.forEach(element => {
        if (element.chartId == id) {
            element.isSelected = true;
            chartCollectionViewModel.currentId = element.chartId;
        }
        else {
            element.isSelected = false;
        }
    });
    chartCollectionViewModel.$nextTick(() => {
        resizeEverything();
    });
}

function getChartViewModelById(id)
{
    var requestViewModel = null;
    chartCollectionViewModel.chartViewModels.forEach(element => {
        if (element.chartId == id) {
            requestViewModel = element;
        }
    });

    return requestViewModel;
}

function onPlayButtonClicked()
{
    currentViewModel = getChartViewModelById(chartCollectionViewModel.currentId);
    if(currentViewModel.data != null)
    {
        playAnimation(currentViewModel.data);
    }
}

function playAnimation(data)
{
    translateData = data;
    startTime = Date.now();
}
