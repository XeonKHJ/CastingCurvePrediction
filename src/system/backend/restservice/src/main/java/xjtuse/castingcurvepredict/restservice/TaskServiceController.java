package xjtuse.castingcurvepredict.restservice;

import java.util.ArrayList;
import java.util.List;

import org.apache.ibatis.exceptions.PersistenceException;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import xjtuse.castingcurvepredict.castingpredictiors.IStatusManager;
import xjtuse.castingcurvepredict.castingpredictiors.TaskStatus;
import xjtuse.castingcurvepredict.data.MlModelMapper;
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
        var taskModels = RestserviceApplication.getConfig().getTasks();
        ArrayList<TaskViewModel> modelViewModels = new ArrayList<TaskViewModel>();
        for (TaskModel model : taskModels) {
            TaskViewModel mlViewModel = new TaskViewModel(model);
            modelViewModels.add(mlViewModel);
        }

        return new TaskCollectionViewModel(modelViewModels);
    }

    @GetMapping("/getStatusByTaskId")
    public TaskStatusViewModel getStatusByTaskId(@RequestParam(value="taskId") int taskId)
    {
        TaskStatusViewModel vm = new TaskStatusViewModel();
        
        var sm = RestserviceApplication.getConfig().getStatusManager(new TaskModel(taskId));
        // vm.setEpochs(sm.readEpochs());
        vm.setLosses(sm.readLosses());
        return vm;
    }

    @GetMapping("/startTrainingTask")
    public StatusViewModel startTrainingTask(@RequestParam(value = "taskId") int id) {
        System.out.println("StartTrainningTask" + id);
        StatusViewModel vm = new StatusViewModel();

        IStatusManager sm = RestserviceApplication.getConfig().getStatusManager(id);
        sm.saveStatus(TaskStatus.Running);
        sm.getTask().Start(RestserviceApplication.getConfig().getCastingGenerator());

        System.out.println("StartTrainningTask" + id + " return");
        return vm;
    }


    @GetMapping("/stopTaskById")
    public StatusViewModel stopTaskById(@RequestParam("taskId") int id) {
        SqlSessionFactory sessionFactory = RestserviceApplication.getSqlSessionFactory();
        StatusViewModel viewModel = new StatusViewModel();
        RestserviceApplication.getConfig().getStatusManager(id).getTask().Stop();
        return viewModel;
    }

    
    @GetMapping("/createTrainingTaskFromModelId")
    public TaskViewModel createTrainingTaskFromModelId(@RequestParam(value = "modelId") int id) {
        TaskViewModel viewModel = null;
        TaskModel task;
        var sessionFactory = RestserviceApplication.getSqlSessionFactory();
        try(var session = sessionFactory.openSession()) {
            // 验证model是否存在。
            var mlModelMapper = session.getMapper(MlModelMapper.class);
            var mlModel = mlModelMapper.getMlModelById(id);

            if(mlModel == null)
            {
                throw  new NullPointerException("学习模型不存在");
            }

            task = new TaskModel(RestserviceApplication.getConfig().generateTaskId(), id);
            task.setLoss(mlModel.getLoss());
            viewModel = new TaskViewModel(task);
            var sm = RestserviceApplication.getConfig().getStatusManager(task);
        } catch (IndexOutOfBoundsException e) {
            viewModel = new TaskViewModel(-1, e.getMessage());
        } catch(NullPointerException e){
            viewModel = new TaskViewModel(-2, e.getMessage());
        }
        
        return viewModel;
    }
}
