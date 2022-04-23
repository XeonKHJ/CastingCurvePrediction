function onLoad() {
    displayLossChart(null)
    taskId = parseInt(getQueryString("taskId"));
    getTaskByid(taskId);
    getTaskStatusById(taskId);
}

window.onresize = function(){
    resizeEverything();
}

var taskStatusVm = Vue.createApp({
    data() {
        return {
            id: 0,
            status: "training",
            losses: {
                time:[],
                value:[]
            }
        }
    }
}).mount("#taskListTable");

function getQueryString(name) {
    let reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)", "i");
    let r = window.location.search.substr(1).match(reg);
    if (r != null) {
        return decodeURIComponent(r[2]);
    };
    return null;
}

function getTaskByid(id) {
    var result = null;
    axios.get(baseServerAddr + '/getTaskById?id=' + id, {
        Headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            Accept: "application/json"
        }
    }).then(response => {
        switch (response.data.statusCode) {
            case 1:
                result = response.data;
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
    return result;
}

function getTaskStatusById(id) {
    axios.get(baseServerAddr + '/getTaskStatusById?id=' + id, {
        Headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            Accept: "application/json"
        }
    }).then(response => {
        data = response.data;

    }).then().catch(
        err => {
            console.log(err)
            showError(err)
        })
}

function displayLossChart(data)
{
    var echart = echarts.init(document.getElementById("lossChartDiv"));
    currentChart = echart
    var option = {
        xAxis: {
            type: 'category',
            data: [1,2,3]
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
                data: [1,2,3],
                yAxis: 0,
                type: 'line'
            },
            {
                data: [1,1.1,1.2],
                yAxis: 2,
                type: 'line'
            }
        ]
    };

    echart.setOption(option)
}

window.setInterval(updateStatus, 2000);

function updateStatus()
{
    const taskStatusAndLoss = getTaskStatusById(id)
    taskStatusAndLoss.losses.forEach(element => {
        taskStatusVm.losses.time.add(losses.time);
        taskStatusVm.losses.value.add(losses.value);
    });
}

function resizeEverything()
{
    echart.resize();
}