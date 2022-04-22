package xjtuse.castingcurvepredict.restservice;

import java.util.ArrayList;
import java.util.List;

import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import xjtuse.castingcurvepredict.castingpredictiors.IStatusManager;
import xjtuse.castingcurvepredict.castingpredictiors.TaskStatus;
import xjtuse.castingcurvepredict.data.TrainTask;
import xjtuse.castingcurvepredict.data.TrainTaskMapper;
import xjtuse.castingcurvepredict.models.TaskModel;
import xjtuse.castingcurvepredict.viewmodels.StatusViewModel;
import xjtuse.castingcurvepredict.viewmodels.TaskCollectionViewModel;
import xjtuse.castingcurvepredict.viewmodels.TaskStatusViewModel;
import xjtuse.castingcurvepredict.viewmodels.TaskViewModel;

@CrossOrigin
@RestController
public class TaskServiceController {
    @GetMapping("/getTasks")
    public TaskCollectionViewModel getTasks() {
        SqlSessionFactory sessionFactory = RestserviceApplication.getSqlSessionFactory();
        List<TrainTask> models = null;
        ArrayList<TaskViewModel> modelViewModels = new ArrayList<TaskViewModel>();
        try (SqlSession session = sessionFactory.openSession()) {
            TrainTaskMapper mlModelMapper = session.getMapper(TrainTaskMapper.class);
            models = mlModelMapper.getTasks();
        }

        if (models != null) {
            for (int i = 0; i < models.size(); ++i) {
                TrainTask model = models.get(i);
                TaskViewModel mlViewModel = new TaskViewModel(model.getId(), model.getLoss(), model.getStatus(), model.getEpoch(), model.getModelId());
                modelViewModels.add(mlViewModel);
            }
        }

        return new TaskCollectionViewModel(modelViewModels);
    }

    @GetMapping("/getTaskStatus")
    public TaskStatusViewModel getStatusByTaskId(@RequestParam(value="taskId") int taskId)
    {
        TaskStatusViewModel vm = new TaskStatusViewModel();
        
        var sm = RestserviceApplication.getConfig().getStatusManager(new TaskModel(taskId));
        vm.setEpochs(sm.readEpochs());
        vm.setLosses(sm.readLosses());

        return vm;
    }

    @GetMapping("/startTrainingTask")
    public StatusViewModel startTrainingTask(@RequestParam(value = "taskId") int id) {
        StatusViewModel vm = new StatusViewModel();
        SqlSessionFactory sessionFactory = RestserviceApplication.getSqlSessionFactory();

        try (SqlSession session = sessionFactory.openSession()) {
            TrainTaskMapper mapper = session.getMapper(TrainTaskMapper.class);
            TrainTask task = mapper.getTaskById(id);
            var taskInstance = task.getInstance();
            IStatusManager sm = RestserviceApplication.getConfig().getStatusManager(taskInstance);
            sm.saveStatus(TaskStatus.Training);
            taskInstance.Start();
        }
        catch(Exception ex){
            vm.setStatusCode(-3);
            vm.setMessage(ex.getMessage());
        }

        return vm;
    }

    @GetMapping("/StopTask")
    public StatusViewModel stopTask(@RequestParam(value = "taskId") int id)
    {
        StatusViewModel vm = new StatusViewModel();
        // TODO 实现停止任务功能.
        return vm;
    }
}
