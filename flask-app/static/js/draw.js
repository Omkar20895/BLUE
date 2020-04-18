
function get_random_grid(){
  $.ajax({
    type: "GET",
    url: "/newgame",
    //data: data,
    success: function( data ){
      console.log(data);
      draw_grid(data.size, data.states);
    },
    dataType: 'json'
  });
}

function register_mode(coord){
  
}

function draw_grid(size, states){
  var marginRight = 5/size[0];
  var marginTop = 5/size[1];
  var cellWidth = 90/size[0];
  var cellHeight = 90/size[1];  
  
  for(i=0;i<size[0];i++){
    for(j=0;j<size[1];j++){
      var sq = $("#container").append("<div class='square' coord="+i+","+j+"></div>");
    }
  }

  $(".square").css("width", cellWidth+"%");
  $(".square").css("height", cellHeight+"%");
  $(".square").css("margin", marginTop+"% "+marginRight+"%");

  update_grid(states);
}

function update_grid(states){
  console.log(states);
  var cell;
  for(var k=0;k<states.length;k++){
    cell = states[k];
    if(cell[2] == 0){
      console.log(0);
      $(".square[coord='"+cell[0].toString()+","+cell[1].toString()+"']").removeClass("affected");
      $(".square[coord='"+cell[0].toString()+","+cell[1].toString()+"']").removeClass("attacked-default");
      $(".square[coord='"+cell[0].toString()+","+cell[1].toString()+"']").removeClass("attacked");
    }
    if(cell[2] == 1){
      console.log(1);
      $(".square[coord='"+cell[0].toString()+","+cell[1].toString()+"']").removeClass("attacked-default");
      $(".square[coord='"+cell[0].toString()+","+cell[1].toString()+"']").addClass("affected");
      $(".square[coord='"+cell[0].toString()+","+cell[1].toString()+"']").removeClass("attacked");
    }
    if(cell[2] == 2){
      console.log(2);
      $(".square[coord='"+cell[0].toString()+","+cell[1].toString()+"']").removeClass("affected");
      $(".square[coord='"+cell[0].toString()+","+cell[1].toString()+"']").addClass("attacked-default");
    }
    if(cell[2] == 3){
      console.log(3);
      $(".square[coord='"+cell[0].toString()+","+cell[1].toString()+"']").removeClass("affected");
      $(".square[coord='"+cell[0].toString()+","+cell[1].toString()+"']").addClass("attacked-default");
    }
  }  
}


function draw_grid_old(){
  var L = 25;
  var p = 0.38;
  var mode = 0;
  var zero_thresh = 0.05;

  var size = L;
  var squaremargin = 10/size;
  var squarewidth = 90/size;
  var count = 0;
  var attackList = [];
  var cutoffs = [];

  for(i=0;i<size;i++){
    for(j=0;j<size;j++){
      var sq = $("#container").append("<div class='square' index='"+count+"'></div>");
      count++;
    }
  }

  $(".square").css("width", squarewidth+"%");
  $(".square").css("height", squarewidth+"%");
  $(".square").css("margin", squaremargin/2+"%");
}
//init();

/*
$(".square").click(function(){
  //$(this).css("background-color","black").fadeIn();
  $(this).addClass("attacked");
  attackList = $(".square.attacked");
  var idx = $(this).attr("index");
  idx = parseInt(idx,10);
  cutoffs = cutoffs.concat(bp.attack_index(idx));
  for(i = 0; i<cutoffs.length; i++){
    $(".square[index='"+cutoffs[i].toString()+"']").addClass("affected");
  }
  //  var idx = parseInt(attackList[i].attributes.index.nodeValue,10);
    //cutoffs = attack_index(idx);
  //}
});


function init(){
  for(var key=0; key< bp.N; key++){
    if(!bp.alive[key]){
      $(".square[index='"+key+"']").addClass("attacked-default");
    }
  }
  bp.get_gcc_membership_vector();
  var cutoff = get_cutoff_ids(bp.status);
  for(i = 0; i<cutoff.length; i++){
    $(".square[index='"+cutoff[i].toString()+"']").addClass("affected");
  }
}
*/