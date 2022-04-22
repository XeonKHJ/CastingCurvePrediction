package xjtuse.castingcurvepredict.restservice;

import java.util.ArrayList;
import java.util.List;

import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import xjtuse.castingcurvepredict.data.TrainTask;
import xjtuse.castingcurvepredict.data.TrainTaskMapper;
import xjtuse.castingcurvepredict.models.TaskModel;
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
    public TaskStatusViewModel getStatusByTaskId(@Param(value="taskId") int taskId)
    {
        TaskStatusViewModel vm = new TaskStatusViewModel();
        
        var sm = RestserviceApplication.getConfig().getStatusManager(new TaskModel(taskId));
        vm.setEpochs(sm.readEpochs());
        vm.setLosses(sm.readLosses());

        return vm;
    }
}
