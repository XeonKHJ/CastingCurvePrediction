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

    @PostMapping("/predictCastingCurve")
    public DiagramViewModel PredictCastingCurve(InputViewModel input)
    {
        ICastingGenerator generator = new ConstCastingGenerator();
        CastingModel castingModel = new CastingModel(generator);
        CastingResultModel resultModel = castingModel.PredictCastingCurve(null);
        return new DiagramViewModel(resultModel);
    }

    @PostMapping("/openCastingCurveFile")
    public DiagramViewModel openCastingCurveFile(MultipartFile file)
    {
        if(file == null)
        {
            DiagramViewModel viewModel = new DiagramViewModel(null);
            viewModel.setStatusCode(-1);
            return viewModel;
        }
        ICastingGenerator generator = new JsonFileCastingGenerator();
        CastingResultModel resultModel = new CastingResultModel();
        File castFile = null;
        try {
            castFile = File.createTempFile("castingcurve", "data");
            file.transferTo(castFile);

            CastingInputModel inputModel = new CastingInputModel();
            inputModel.getKeyValues().put("file", castFile);
            generator.PredcitCastingCurve(inputModel);
            CastingModel castingModel = new CastingModel(generator);
            resultModel = castingModel.PredictCastingCurve(inputModel);
        } catch (IOException e) {
            resultModel.setStatusCode(-2);
        }

        if(castFile != null)
        {
            castFile.delete();
        }
        

        return new DiagramViewModel(resultModel);
    }

    HashMap<UUID, TrainningStatusViewModel> uploadList = new HashMap<UUID, TrainningStatusViewModel>();
    @PostMapping("/uploadAndTrainModel")
    public UploadResultViewModel uploadAndTrainModel(MultipartFile file)
    {
        UploadResultViewModel resultViewModel = new UploadResultViewModel();
        if (file.isEmpty()) {
            resultViewModel.setStatusCode(-1);
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
        } catch (IOException e) {
            resultViewModel.setStatusCode(-2);
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
            statusViewModel.setStatusCode(-3);
        }

        return statusViewModel;
    }

    // @GetMapping("/exportCastingCurveData")
    // public StreamingResponseBody exportCastingCurveData()
    // {

    // }

    // public StreamingResponseBody downloadFile(HttpServletResponse response, @PathVariable Long fileId) {

    //     FileInfo fileInfo = fileService.findFileInfo(fileId);
    //     response.setContentType(fileInfo.getContentType());
    //     response.setHeader(
    //         HttpHeaders.CONTENT_DISPOSITION, "attachment;filename=\"" + fileInfo.getFilename() + "\"");
    
    //     return outputStream -> {
    //         int bytesRead;
    //         byte[] buffer = new byte[BUFFER_SIZE];
    //         InputStream inputStream = fileInfo.getInputStream();
    //         while ((bytesRead = inputStream.read(buffer)) != -1) {
    //             outputStream.write(buffer, 0, bytesRead);
    //         }
    //     };
    // }
}
