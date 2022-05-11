package xjtuse.castingcurvepredict.restservice;

import java.io.ObjectInputFilter.Status;
import java.time.Instant;
import java.util.ArrayList;

import java.util.List;
import org.apache.ibatis.exceptions.PersistenceException;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.yaml.snakeyaml.constructor.DuplicateKeyException;

import xjtuse.castingcurvepredict.config.IConfigFactory;
import xjtuse.castingcurvepredict.data.MlModel;
import xjtuse.castingcurvepredict.data.MlModelMapper;
import xjtuse.castingcurvepredict.models.TaskModel;
import xjtuse.castingcurvepredict.utils.utils;
import xjtuse.castingcurvepredict.viewmodels.MLModelViewModel;
import xjtuse.castingcurvepredict.viewmodels.ModelCollectionViewModel;
import xjtuse.castingcurvepredict.viewmodels.StatusViewModel;
import xjtuse.castingcurvepredict.viewmodels.TaskViewModel;

// 所有从模型开始的请求都在这个控制器中定义
@CrossOrigin
@RestController
public class LearningModelServiceConstoller {
    @GetMapping("/getModelFromId")
    public MLModelViewModel getModelFromId(@RequestParam(value = "id") int id) {
        SqlSessionFactory sessionFactory = RestserviceApplication.getSqlSessionFactory();

        try (SqlSession session = sessionFactory.openSession()) {
            MlModelMapper mlModelMapper = session.getMapper(MlModelMapper.class);
            MlModel model = mlModelMapper.getMlModelById(1);
        }

        return null;
    }

    @GetMapping("/getModels")
    public ModelCollectionViewModel getModels() {
        SqlSessionFactory sessionFactory = RestserviceApplication.getSqlSessionFactory();
        List<MlModel> models = null;
        ModelCollectionViewModel collectionViewModel = new ModelCollectionViewModel(new ArrayList<MLModelViewModel>());
        ArrayList<MLModelViewModel> modelViewModels = new ArrayList<MLModelViewModel>();
        try (SqlSession session = sessionFactory.openSession()) {
            MlModelMapper mlModelMapper = session.getMapper(MlModelMapper.class);
            models = mlModelMapper.getModels();
        }

        if (models == null) {

        } else {
            for (int i = 0; i < models.size(); ++i) {
                MLModelViewModel mlViewModel = new MLModelViewModel(models.get(i));
                modelViewModels.add(mlViewModel);
                collectionViewModel = new ModelCollectionViewModel(modelViewModels);
            }
        }

        return collectionViewModel;
    }

    @GetMapping("/deleteModelById")
    public StatusViewModel deleteModelById(@RequestParam(value = "id") int id) {
        SqlSessionFactory sessionFactory = RestserviceApplication.getSqlSessionFactory();
        StatusViewModel viewModel = new StatusViewModel();

        try (SqlSession session = sessionFactory.openSession()) {
            if(RestserviceApplication.getConfig().getTaskFromModelId(id) == null)
            {
                MlModelMapper mapper = session.getMapper(MlModelMapper.class);
                mapper.deleteModelById(id);
                session.commit();
            }
            else
            {
                throw new PersistenceException("有任务正在运行");
            }
        } catch (PersistenceException exception) {
            viewModel.setStatusCode(-100);
            viewModel.setMessage("有任务正在运行");
        }

        return viewModel;
    }

    @GetMapping("/createModel")
    public MLModelViewModel createModel() {
        SqlSessionFactory sessionFactory = RestserviceApplication.getSqlSessionFactory();
        MLModelViewModel viewModel = null;
        try (SqlSession session = sessionFactory.openSession()) {
            var mapper = session.getMapper(MlModelMapper.class);
            MlModel model = new MlModel();
            model.setName("model-" + Instant.now().toString());
            model.setPath("C:\\model-" + Instant.now().toString() + ".model");
            mapper.createModel(model);
            session.commit();
            viewModel = new MLModelViewModel(model);
        }

        return viewModel;
    }
}
