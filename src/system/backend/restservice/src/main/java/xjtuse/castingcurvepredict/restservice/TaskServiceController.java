package xjtuse.castingcurvepredict.restservice;

import java.util.ArrayList;
import java.util.List;

import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import xjtuse.castingcurvepredict.data.TrainTask;
import xjtuse.castingcurvepredict.data.TrainTaskMapper;
import xjtuse.castingcurvepredict.viewmodels.TaskCollectionViewModel;
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
            models = mlModelMapper.getModels();
        }

        if(models == null)
        {
            
        }
        else
        {
            for(int i = 0; i < models.size(); ++i)
            {
                TaskViewModel mlViewModel = new TaskViewModel(models.get(i).getId(), models.get(i).getLoss(), models.get(i).getStatus());
                modelViewModels.add(mlViewModel);
            }
        }

        return new TaskCollectionViewModel(modelViewModels);
    }
}
