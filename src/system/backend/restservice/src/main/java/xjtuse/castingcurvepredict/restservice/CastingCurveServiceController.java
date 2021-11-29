package xjtuse.castingcurvepredict.restservice;

import java.io.File;
import java.io.IOException;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import xjtuse.castingcurvepredict.castingpredictiors.ConstCastingGenerator;
import xjtuse.castingcurvepredict.interfaces.ICastingGenerator;
import xjtuse.castingcurvepredict.models.CastingModel;
import xjtuse.castingcurvepredict.models.CastingResultModel;
import xjtuse.castingcurvepredict.viewmodels.DiagramViewModel;
import xjtuse.castingcurvepredict.viewmodels.UploadResultViewModel;

@CrossOrigin
@RestController
public class CastingCurveServiceController {
    @GetMapping("/getcastingcurvefrominput")
    public DiagramViewModel GetCastingCurveFromInput() {
        // Construct a diagram view model;
        ICastingGenerator generator = new ConstCastingGenerator();

        // Stab
        CastingModel castingModel = new CastingModel(generator);

        CastingResultModel resultModel = castingModel.PredictCastingCurve(null);

        return new DiagramViewModel(resultModel);
    }

    @PostMapping("/uploadAndTrainModel")
    public UploadResultViewModel uploadAndTrainModel(MultipartFile file)
    {
        UploadResultViewModel resultViewModel = new UploadResultViewModel();
        if (file.isEmpty()) {
            resultViewModel.setStatus("文件为空。");
        }

        String fileName = file.getOriginalFilename();
        String filePath = CastingConfig.getModelFolderPath();
        File dest = new File(filePath + fileName);
        try {
            file.transferTo(dest);
            resultViewModel.setStatus("上传成功");
        } catch (IOException e) {
            resultViewModel.setStatus("上传失败");
        }
        return resultViewModel;
    }
}
