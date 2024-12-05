$(document).ready(function(){

    // Set initial state
    $('.video-container').hide();
    $('.ROI_sliders').hide();
    $('.ROI_DisplayCornerValues').hide();
    $('.Apply_Mask').hide();
    $('.Video').attr('src', '/raw_video');
    $('.video-container').show();

    $('.nav-link').click(function () {
        var caption = $(this).data('caption');
        console.log(caption);

        $('.video-container').hide();
        $('.CornerTarget').hide();
        $('.ROI_sliders').hide();

        if (caption === "Original Video") {
            $('.Video').attr('src', '/raw_video');
            $('.ROI_DisplayCornerValues').hide();
            $('.Apply_Mask').hide();
            $('.video-container').show();
        } 
        else if (caption === "Select ROI") {
            $('.Video').attr('src', '/streched_video');
            $('.video-container').show();
            $('.CornerTarget').show();
            $('.ROI_DisplayCornerValues').show();
            $('.Apply_Mask').hide();
        } 
        else if (caption === "Set HSV Values") {
            $('.Video').attr('src', '/mask_video');
            $('.video-container').show();
            $('.CornerTarget').hide();
            $('.ROI_DisplayCornerValues').hide();
            $('.ROI_sliders').show();
            $('.Apply_Mask').show();
        }
        else if (caption === "Detect Object with color") {
            $('.Video').attr('src', '/ObjectDetectionWithColor');
            $('.video-container').show();
            $('.CornerTarget').hide();
            $('.ROI_DisplayCornerValues').hide();
            $('.ROI_sliders').hide();
            $('.Apply_Mask').hide();
        }
        else if (caption === "Detect Faces") {
            $('.Video').attr('src', '/DetectFaces');
            $('.video-container').show();
            $('.CornerTarget').hide();
            $('.ROI_DisplayCornerValues').hide();
            $('.ROI_sliders').hide();
            $('.Apply_Mask').hide();
        }
        else if (caption === "Hand Gesture Recognition") {
            $('.Video').attr('src', '/HandGestureRecognition');
            $('.video-container').show();
            $('.CornerTarget').hide();
            $('.ROI_DisplayCornerValues').hide();
            $('.ROI_sliders').hide();
            $('.Apply_Mask').hide();
        }
    });

    $(".set-roi").click(function() {
        updateAndSendPositions();
    });

    $(".clear-roi").click(function () {
        console.log(`Reset`)
        $.ajax({ type: "POST", contentType: "application/json", url: "/ResetFieldOfView" });
    });

    $(".SliderHSV").on("input",function () {
        $(this).prev("span").text($(this).val());
        Values =
        {
          "LowerH": $("#SliderLowerH").val(),
          "LowerS": $("#SliderLowerS").val(),
          "LowerV": $("#SliderLowerV").val(),
          "UpperH": $("#SliderUpperH").val(),
          "UpperS": $("#SliderUpperS").val(),
          "UpperV": $("#SliderUpperV").val()
        };
    
        $.ajax({ type: "POST", contentType: "application/json", url: "/UpdateHSV", data: JSON.stringify(Values) });
      });

    // animations
    function test(){
        var tabsNewAnim = $('#navbarSupportedContent');
        var selectorNewAnim = $('#navbarSupportedContent').find('li').length;
        var activeItemNewAnim = tabsNewAnim.find('.active');
        var activeWidthNewAnimHeight = activeItemNewAnim.innerHeight();
        var activeWidthNewAnimWidth = activeItemNewAnim.innerWidth();
        var itemPosNewAnimTop = activeItemNewAnim.position();
        var itemPosNewAnimLeft = activeItemNewAnim.position();
        $(".hori-selector").css({
            "top": itemPosNewAnimTop.top + "px", 
            "left": itemPosNewAnimLeft.left + "px",
            "height": activeWidthNewAnimHeight + "px",
            "width": activeWidthNewAnimWidth + "px"
        });
        $("#navbarSupportedContent").on("click", "li", function(e){
            $('#navbarSupportedContent ul li').removeClass("active");
            $(this).addClass('active');
            var activeWidthNewAnimHeight = $(this).innerHeight();
            var activeWidthNewAnimWidth = $(this).innerWidth();
            var itemPosNewAnimTop = $(this).position();
            var itemPosNewAnimLeft = $(this).position();
            $(".hori-selector").css({
                "top": itemPosNewAnimTop.top + "px", 
                "left": itemPosNewAnimLeft.left + "px",
                "height": activeWidthNewAnimHeight + "px",
                "width": activeWidthNewAnimWidth + "px"
            });
        });
    }

    $(window).on('resize', function(){
        setTimeout(function(){ test(); }, 500);
    });
    $(".navbar-toggler").click(function(){
        $(".navbar-collapse").slideToggle(300);
        setTimeout(function(){ test(); });
    });

    setTimeout(function(){ test(); });

    const container = document.querySelector('.video-container');
    const targets = document.querySelectorAll('.CornerTarget');

    targets.forEach(target => {
        target.addEventListener('mousedown', function(e) {
            let isDragging = true;
            const startX = e.clientX;
            const startY = e.clientY;
            const rect = target.getBoundingClientRect();
            const offsetX = startX - rect.left;
            const offsetY = startY - rect.top;

            const onPointerMove = (e) => {
                if (isDragging) {
                    requestAnimationFrame(() => {
                        const mouseX = e.clientX;
                        const mouseY = e.clientY;
                        const containerRect = container.getBoundingClientRect();
                        const targetRect = target.getBoundingClientRect();

                        let newLeft = mouseX - containerRect.left - offsetX;
                        let newTop = mouseY - containerRect.top - offsetY;

                        // Constrain the target within the container
                        newLeft = Math.max(0, Math.min(newLeft, containerRect.width - targetRect.width));
                        newTop = Math.max(0, Math.min(newTop, containerRect.height - targetRect.height));

                        target.style.left = newLeft + 'px';
                        target.style.top = newTop + 'px';

                        // // Update positions and send to backend
                        // updateAndSendPositions();
                    });
                }
            };

            const onPointerUp = () => {
                isDragging = false;
                document.removeEventListener('mousemove', onPointerMove);
                document.removeEventListener('mouseup', onPointerUp);
            };

            document.addEventListener('mousemove', onPointerMove);
            document.addEventListener('mouseup', onPointerUp);
        });
    });

    function updateAndSendPositions() {
        const CornerPoints = [
            getCornerPosition('P0'),
            getCornerPosition('P1'),
            getCornerPosition('P2'),
            getCornerPosition('P3')
        ];

        const Points = {
            "X0": CornerPoints[0].x, "Y0": CornerPoints[0].y,
            "X1": CornerPoints[1].x, "Y1": CornerPoints[1].y,
            "X2": CornerPoints[2].x, "Y2": CornerPoints[2].y,
            "X3": CornerPoints[3].x, "Y3": CornerPoints[3].y
        };

        // Update input fields
        document.querySelector('.CornerTL').value = `${Math.floor(Points.X0)}, ${Math.floor(Points.Y0)}`;
        document.querySelector('.CornerTR').value = `${Math.floor(Points.X1)}, ${Math.floor(Points.Y1)}`;
        document.querySelector('.CornerBL').value = `${Math.floor(Points.X2)}, ${Math.floor(Points.Y2)}`;
        document.querySelector('.CornerBR').value = `${Math.floor(Points.X3)}, ${Math.floor(Points.Y3)}`;

        console.log("Sending data:", Points);

        $.ajax({
            type: "POST",
            contentType: "application/json",
            url: "/SetFieldOfView",
            data: JSON.stringify(Points),
            success: function (response) {
                console.log("Server response:", response);
            },
            error: function (xhr, status, error) {
                console.error("Error sending data:", error);
                console.log("XHR object:", xhr);
            }
        });
    }

    function getCornerPosition(id) {
        const corner = document.getElementById(id);
        if (!corner) {
            console.error(`Element with id ${id} not found`);
            return { x: 0, y: 0 };
        }
        const rect = corner.getBoundingClientRect();
        const containerRect = container.getBoundingClientRect();
        return {
            x: rect.left- 8 - containerRect.left  + rect.width  / 2,
            y: rect.top - 8 - containerRect.top + rect.height / 2
        };
    }

});
