package xjtuse.castingcurvepredict.restservice;

import java.util.ArrayList;
import java.util.List;

import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import xjtuse.castingcurvepredict.data.MlModelMapper;
import xjtuse.castingcurvepredict.data.Project;
import xjtuse.castingcurvepredict.viewmodels.ProjectCollectionViewModel;
import xjtuse.castingcurvepredict.viewmodels.ProjectViewModel;

@CrossOrigin
@RestController
public class ProjectController {
    @GetMapping("/getProjects")
    public ProjectCollectionViewModel getProjects() {
        SqlSessionFactory sessionFactory = RestserviceApplication.getSqlSessionFactory();
        List<Project> models = null;
        ProjectCollectionViewModel collectionViewModel = new ProjectCollectionViewModel(new ArrayList<ProjectViewModel>());
        ArrayList<ProjectViewModel> modelViewModels = new ArrayList<ProjectViewModel>();
        try (SqlSession session = sessionFactory.openSession()) {
            MlModelMapper mlModelMapper = session.getMapper(MlModelMapper.class);
            models = mlModelMapper.getProjects();
        }

        if (models == null) {

        } else {
            for (int i = 0; i < models.size(); ++i) {
                ProjectViewModel mlViewModel = new ProjectViewModel(models.get(i));
                modelViewModels.add(mlViewModel);
                collectionViewModel = new ProjectCollectionViewModel(modelViewModels);
            }
        }

        return collectionViewModel;
    }
}
