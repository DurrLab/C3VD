$("#slider").on("input change", (e)=>{
    const sliderPos = e.target.value;
    // Update the width of the foreground image
    $('.foreground-img').css('width', `${sliderPos}%`)
    // Update the position of the slider button
    $('.slider-button').css('left', `calc(${sliderPos}% - 18px)`)
});

$(".btn.depth").on('click', function(event){
    $(".btn.depth").addClass('focus')
    $(".btn.normals").removeClass('focus')
    $(".btn.occlusion").removeClass('focus')
    $(".btn.flow").removeClass('focus')
    $('.background-img').css('background-image', 'url("./assets/img/slider/depth.png")')    
});
$(".btn.normals").on('click', function(event){
    $(".btn.depth").removeClass('focus')
    $(".btn.normals").addClass('focus')
    $(".btn.occlusion").removeClass('focus')
    $(".btn.flow").removeClass('focus')
    $('.background-img').css('background-image', 'url("./assets/img/slider/normals.png")')    
});
$(".btn.occlusion").on('click', function(event){
    $(".btn.depth").removeClass('focus')
    $(".btn.normals").removeClass('focus')
    $(".btn.occlusion").addClass('focus')
    $(".btn.flow").removeClass('focus')
    $('.background-img').css('background-image', 'url("./assets/img/slider/occlusion.png")')    
});
$(".btn.flow").on('click', function(event){
    $(".btn.depth").removeClass('focus')
    $(".btn.normals").removeClass('focus')
    $(".btn.occlusion").removeClass('focus')
    $(".btn.flow").addClass('focus')
    $('.background-img').css('background-image', 'url("./assets/img/slider/flow.png")')    
});