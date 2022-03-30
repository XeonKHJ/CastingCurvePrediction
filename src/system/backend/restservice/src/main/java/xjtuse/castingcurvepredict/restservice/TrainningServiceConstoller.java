package xjtuse.castingcurvepredict.restservice;

import java.util.ArrayList;

import org.apache.ibatis.builder.IncompleteElementException;
import org.apache.ibatis.session.SqlSession;
import org.apache.ibatis.session.SqlSessionFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import xjtuse.castingcurvepredict.viewmodels.MLModelViewModel;

//import xjtuse.castingcurvepredict.data.MlModelMapper;

@RestController
public class TrainningServiceConstoller {
    @GetMapping("/getcastingcurvefrominput")
    public MLModelViewModel getModelFromId(@RequestParam(value = "id") int id) {
        SqlSessionFactory sessionFactory = RestserviceApplication.getSqlSessionFactory();
        // MlModelMapper mapper = session.getMapper(MlModelMapper.class);
        
        MLModelViewModel model = new MLModelViewModel();

        return model;
    }

    public ArrayList<MLModelViewModel> getModels()
    {
        throw new IncompleteElementException();
    }
}
