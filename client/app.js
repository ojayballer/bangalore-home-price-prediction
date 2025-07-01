function onPageLoad() {
    console.log("document loaded");

    var locationUrl = "http://127.0.0.1:5000/get_location_names";
    var areaTypeUrl = "http://127.0.0.1:5000/get_area_types";

    $.get(locationUrl, function(data, status) {
        console.log("got response for get_location_names request");
        if (data) {
            var locations = data.locations;
            $('#uiLocations').empty();
            for (var i in locations) {
                var opt = new Option(locations[i]);
                $('#uiLocations').append(opt);
            }
        }
    });

    
    $.get(areaTypeUrl, function(data, status) {
        console.log("got response for get_area_types request");
        if (data) {
            var areaTypes = data.area_types;
            $('#uiArea').empty();
            for (var i in areaTypes) {
                var opt = new Option(areaTypes[i]);
                $('#uiArea').append(opt);
            }
        }
    });
}

function onClickedEstimatePrice() {
    console.log("Estimate price button clicked");

    var sqft = parseFloat(document.getElementById("uiSqft").value);
    var bedrooms = document.querySelector('input[name="uiBedrooms"]:checked').value;
    var bathrooms = document.querySelector('input[name="uiBathrooms"]:checked').value;
    var location = document.getElementById("uiLocations").value;
    var area_type = document.getElementById("uiArea").value;

    var url = "http://127.0.0.1:5000/predict_home_price";

    $.post(url, {
        total_sqft: sqft,
        bedroom: bedrooms,
        bath: bathrooms,
        location: location,
        area_type: area_type
    }, function(data, status) {
        console.log(data.estimated_price);
        document.getElementById("uiEstimatedPrice").innerHTML = "<h2>" + data.estimated_price.toString() + " Lakhs</h2>";
    });
}

window.onload = onPageLoad;
