package xjtuse.castingcurvepredict.restservice;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import xjtuse.castingcurvepredict.castingpredictiors.IStatusManager;
import xjtuse.castingcurvepredict.castingpredictiors.TaskStatus;
import xjtuse.castingcurvepredict.models.TaskModel;
import xjtuse.castingcurvepredict.viewmodels.StatusViewModel;

@RestController
public class ImplModuleServiceController {
    @GetMapping("/uploadTaskStatus")
    public StatusViewModel uploadTaskStatus(@RequestParam("taskId") int taskId, String status) {
        // TODO 获取任务模型
        TaskModel model = new TaskModel(12);
        StatusViewModel vm = new StatusViewModel();
        IStatusManager sm = RestserviceApplication.getConfig().getStatusManager(model);

        if (status.equals("Training")) {
            sm.saveStatus(TaskStatus.Training);
        } else if (status.equals("Stopped")) {
            sm.saveStatus(TaskStatus.Stopped);
        } else if (status.equals("Updating")) {
            sm.saveStatus(TaskStatus.Updating);
        } else if (status.equals("Stopping")) {
            // TODO Stop the task.

            // Stop the task
            sm.saveStatus(TaskStatus.Stopped);
                
            // take it out of the manager.

            // update database.
        }

        // TODO 返回状态
        vm.setStatusCode(1);

        return vm;
    }
}
