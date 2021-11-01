package xjtuse.castingcurvepredict.restservice;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import xjtuse.castingcurvepredict.castingpredictiors.ConstCastingGenerator;
import xjtuse.castingcurvepredict.interfaces.ICastingGenerator;
import xjtuse.castingcurvepredict.models.CastingModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;
import xjtuse.castingcurvepredict.viewmodels.DiagramViewModel;

@CrossOrigin
@RestController
public class CastingCurveServiceController {
    @GetMapping("/getcastingcurvefrominput")
    public DiagramViewModel GetCastingCurveFromInput()
    {
        // Construct a diagram view model;
        ICastingGenerator generator = new ConstCastingGenerator();

        // Stab
        CastingModel castingModel = new CastingModel(generator);

        CastingResultModel resultModel = castingModel.PredictCastingCurve(null);

        return new DiagramViewModel(resultModel);
    }
}
