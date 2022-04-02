package xjtuse.castingcurvepredict.restservice;

import java.util.ArrayList;
import java.util.List;

import org.apache.ibatis.builder.IncompleteElementException;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import xjtuse.castingcurvepredict.data.MlModel;
import xjtuse.castingcurvepredict.data.MlModelMapper;
import xjtuse.castingcurvepredict.viewmodels.MLModelViewModel;
import xjtuse.castingcurvepredict.viewmodels.ModelCollectionViewModel;

//import xjtuse.castingcurvepredict.data.MlModelMapper;

@CrossOrigin
@RestController
public class TrainningServiceConstoller {
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

        if(models == null)
        {
            
        }
        else
        {
            for(int i = 0; i < models.size(); ++i)
            {
                MLModelViewModel mlViewModel = new MLModelViewModel(models.get(i));
                modelViewModels.add(new MLModelViewModel(models.get(i)));
                collectionViewModel = new ModelCollectionViewModel(modelViewModels);
            }
        }

        return collectionViewModel;
    }
}
