var body = document.getElementsByName("body");
document.createElement()

var chartCollectionViewModel = Vue.createApp({
    data() {
        return {
            isError: false,
            errorMessage:"empty",

        }
    }
}).mount("#chartSection");

function hasError(data)
{
    var hasError = false;
    var errorMessage;
    if(data.statusCode != undefined && data.status != undefined)
    {
        if(data.status > 0)
        {
            hasError = true;
        }
        errorMessage = data.status;
    }

    return hasError;
}

function displayError(data)
{
    var htmlBody = document.getElementsByName("body");
    
    
}

