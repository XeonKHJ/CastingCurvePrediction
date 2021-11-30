package xjtuse.castingcurvepredict.restservice;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.UUID;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import xjtuse.castingcurvepredict.castingpredictiors.*;
import xjtuse.castingcurvepredict.interfaces.*;
import xjtuse.castingcurvepredict.models.*;
import xjtuse.castingcurvepredict.viewmodels.*;

@CrossOrigin
@RestController
public class CastingCurveServiceController {
    @GetMapping("/getcastingcurvefrominput")
    public DiagramViewModel GetCastingCurveFromInput() {
        // Construct a diagram view model;
        ICastingGenerator generator = new ConstCastingGenerator();

        CastingModel castingModel = new CastingModel(generator);

        CastingResultModel resultModel = castingModel.PredictCastingCurve(null);

        return new DiagramViewModel(resultModel);
    }


    HashMap<UUID, TrainningStatusViewModel> uploadList = new HashMap<UUID, TrainningStatusViewModel>();
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

        // check file.
        TrainningStatusViewModel statusViewModel = new TrainningStatusViewModel();
        // Generate UUID.
        UUID uuid = UUID.randomUUID();
        uploadList.put(uuid, statusViewModel);

        try {
            file.transferTo(dest);
            resultViewModel.setStatus("上传成功");
        } catch (IOException e) {
            resultViewModel.setStatus("上传失败");
        }

        ICastingGenerator generator = new ConstCastingGenerator();
        CastingModel castingModel = new CastingModel(generator);        
        CastingResultModel resultModel = castingModel.PredictCastingCurve(null);

        uploadList.remove(uuid);

        return resultViewModel;
    }

    @GetMapping("/getUploadAndTrainningStatus")
    public TrainningStatusViewModel getUploadAndTrainningStatus(@RequestParam(value = "uuid") String uuidString)
    {
        UUID uuid = UUID.fromString(uuidString);
        TrainningStatusViewModel statusViewModel = uploadList.get(uuid);
        if(statusViewModel != null)
        {
            // Check status;
            
        }
        else{
            statusViewModel = new TrainningStatusViewModel();
            statusViewModel.setStatus("查询不到训练活动的状态。");
        }

        return statusViewModel;
    }
}
