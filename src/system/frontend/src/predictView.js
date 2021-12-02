var formVm = Vue.createApp({
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
    axios.get('http://localhost:8080/getcastingcurvefrominput', {
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
        startTime = Date.now();
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