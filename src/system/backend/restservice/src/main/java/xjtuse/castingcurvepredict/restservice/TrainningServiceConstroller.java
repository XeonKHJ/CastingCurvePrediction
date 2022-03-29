package xjtuse.castingcurvepredict.restservice;

import org.apache.ibatis.session.SqlSessionFactory;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import xjtuse.castingcurvepredict.viewmodels.MLModelViewModel;

@RestController
public class TrainningServiceConstroller {
    @GetMapping("/getcastingcurvefrominput")
    public MLModelViewModel getModelFromId(@RequestParam(value = "id") int id) {
        SqlSessionFactory sessionFactory = RestserviceApplication.getSqlSessionFactory();
        
        
        MLModelViewModel model = new MLModelViewModel();

        return model;
    }
}
