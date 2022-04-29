package xjtuse.castingcurvepredict.restservice;

import java.util.Date;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import ch.qos.logback.core.status.Status;
import xjtuse.castingcurvepredict.castingpredictiors.IStatusManager;
import xjtuse.castingcurvepredict.castingpredictiors.TaskStatus;
import xjtuse.castingcurvepredict.data.TrainTask;
import xjtuse.castingcurvepredict.data.TrainTaskMapper;
import xjtuse.castingcurvepredict.inputmodels.UploadTaskLossesInputModel;
import xjtuse.castingcurvepredict.models.TaskModel;
import xjtuse.castingcurvepredict.utils.utils;
import xjtuse.castingcurvepredict.viewmodels.StatusViewModel;

@RestController
public class ImplModuleServiceController {
    @GetMapping("/uploadTaskStatus")
    public StatusViewModel uploadTaskStatus(@RequestParam("taskId") int taskId, String status) {
        System.out.println("uploadTaskStatus");
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
        System.out.println("uploadTaskStatus return");
        return vm;
    }

    @PostMapping("/UploadLosses")
    public StatusViewModel uploadLosses(@RequestBody UploadTaskLossesInputModel payload) throws ParseException {
        System.out.println("UploadLosses");
        StatusViewModel vm = new StatusViewModel();
        int taskId = payload.getTaskId();
        double[] losses = payload.getLosses();
        SqlSessionFactory sessionFactory = RestserviceApplication.getSqlSessionFactory();
        SortedMap<Date, Double> map = new TreeMap<>();
 
        for (int i = 0; i < losses.length; ++i) {
            Date d = utils.stringToDate(payload.getTimes()[i]);
            map.put(d, losses[i]);
        }

        try (SqlSession session = sessionFactory.openSession()) {
            TrainTaskMapper mapper = session.getMapper(TrainTaskMapper.class);
            TrainTask task = mapper.getTaskById(taskId);
            var taskInstance = task.getInstance();
            IStatusManager sm = RestserviceApplication.getConfig().getStatusManager(taskInstance);
            sm.saveLosses(map);
        }
        System.out.println("UploadLosses return");
        return vm;
    }
}