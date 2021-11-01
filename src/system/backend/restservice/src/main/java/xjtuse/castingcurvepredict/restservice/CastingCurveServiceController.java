package xjtuse.castingcurvepredict.restservice;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import xjtuse.castingcurvepredict.viewmodels.DiagramViewModel;

@RestController
public class CastingCurveServiceController {
    @GetMapping("/getcastingcurvefrominput")
    public DiagramViewModel GetCastingCurveFromInput()
    {
        // Construct a diagram view model;
        // Stab
        

        return new DiagramViewModel();
    }
}
