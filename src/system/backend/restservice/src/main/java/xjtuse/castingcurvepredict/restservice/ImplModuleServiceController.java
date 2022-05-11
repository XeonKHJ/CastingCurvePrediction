package xjtuse.castingcurvepredict.restservice;

import java.util.Date;
import java.text.ParseException;

import java.util.SortedMap;
import java.util.TreeMap;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import xjtuse.castingcurvepredict.castingpredictiors.IStatusManager;

import xjtuse.castingcurvepredict.inputmodels.UploadTaskLossesInputModel;
import xjtuse.castingcurvepredict.models.TaskModel;
import xjtuse.castingcurvepredict.utils.utils;
import xjtuse.castingcurvepredict.viewmodels.StatusViewModel;

// 该Controller由实现训练和预测的程序调用，并非由前端调用。
@RestController
public class ImplModuleServiceController {
    @GetMapping("/uploadTaskStatus")
    public StatusViewModel uploadTaskStatus(@RequestParam("taskId") int taskId, String status) {
        System.out.println("uploadTaskStatus");

        TaskModel model = new TaskModel(taskId);
        StatusViewModel vm = new StatusViewModel();
        IStatusManager sm = RestserviceApplication.getConfig().getStatusManager(model);

        sm.saveStatus(utils.StringToTaskStatus(status));

        vm.setStatusCode(1);
        System.out.println("uploadTaskStatus return");
        return vm;
    }

    @PostMapping("/UploadLosses")
    public StatusViewModel uploadLosses(@RequestBody UploadTaskLossesInputModel payload) throws ParseException {
        System.out.println("UploadLosses");
        StatusViewModel vm = new StatusViewModel();
        int taskId = payload.getTaskId();
        double[] losses = payload.getLosses();

        SortedMap<Date, Double> map = new TreeMap<>();
        
        double lastLossValue = Double.MAX_VALUE;
        for (int i = 0; i < losses.length; ++i) {
            Date d = utils.stringToDate(payload.getTimes()[i]);
            map.put(d, losses[i]);
            if(i == losses.length - 1)
            {
                lastLossValue = losses[i];
            }
        }

        IStatusManager sm = RestserviceApplication.getConfig().getStatusManager(taskId);

        if(lastLossValue != Double.MAX_VALUE)
        {
            sm.getTask().setLoss(lastLossValue);
        }
        
        sm.saveLosses(map);

        System.out.println("UploadLosses return");
        return vm;
    }
}
