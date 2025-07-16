/**********************************************************************************/
/* Copyright (c) 2025 Matthew Thomas Beck                                         */
/*                                                                                */
/* All rights reserved. This code and its associated files may not be reproduced, */
/* modified, distributed, or otherwise used, in part or in whole, by any person   */
/* or entity without the express written permission of the copyright holder,      */
/* Matthew Thomas Beck.                                                           */
/**********************************************************************************/





/******************************************************/
/*************** MODIFY GLOBAL ELEMENTS ***************/
/******************************************************/

/********** NAV BAR **********/







/**********************************************************/
/*************** controller.html ANIMATIONS ***************/
/**********************************************************/





/***************************************************************/
/*************** controller.html CUSTOM ELEMENTS ***************/
/***************************************************************/


/********** CUSTOM ELEMENTS **********/

.authBox {
    margin: auto;
    padding: 40px 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    width: 100%;
    text-align: center;
    color: white;
}

.statusBox {
    margin: 20px auto;
    padding: 20px 30px;
    border-radius: 8px;
    font-size: 1.2em;
    width: fit-content;
    text-align: center;
}
.statusBox.success {
    background: #2ecc40;
    color: #fff;
    border: 2px solid #27ae60;
}
.statusBox.denied {
    background: #ff4136;
    color: #fff;
    border: 2px solid #c0392b;
}
.controllerInstructions {
    margin-top: 20px;
    font-size: 1.1em;
    color: #fff;
}
#videoStreamPlaceholder {
    margin: 30px 0;
    padding: 20px;
    background: #222;
    color: #bbb;
    border-radius: 8px;
}

/* Minimal Robot Controller Styles */
#videoContainer {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #000;
    overflow: hidden;
}

#robotVideo {
    width: 100%;
    height: 100%;
    object-fit: cover;
    background: #000;
}

/* Canvas styling for video display */
#videoContainer canvas {
    width: 100vw;
    height: 100vh;
    object-fit: cover;
    border: none;
    border-radius: 0;
    background: #000;
    position: absolute;
    top: 0;
    left: 0;
    z-index: 1;
}

#connectButton {
    position: absolute;
    top: calc(var(--navBarHeight) + 15px);
    right: 20px;
    background: #2ecc40;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 6px;
    cursor: pointer;
    z-index: 10;
    font-size: 16px;
    font-weight: bold;
    transition: background-color 0.3s ease;
}

#connectButton:hover {
    background: #27ae60;
}

#connectButton:disabled {
    background: #666;
    cursor: not-allowed;
}

#leaveButton {
    position: absolute;
    top: calc(var(--navBarHeight) + 55px);
    right: 20px;
    background: #ff4136;
    color: white;
    border: none;
    padding: 12px 24px;
    border-radius: 6px;
    cursor: pointer;
    z-index: 10;
    font-size: 16px;
    font-weight: bold;
    transition: background-color 0.3s ease;
}

#leaveButton:hover {
    background: #c0392b;
}

/* Control instructions overlay */
.controlInstructions {
    position: absolute;
    top: calc(var(--navBarHeight) + 15px);
    left: 20px;
    background: rgba(0, 0, 0, 0.8);
    color: white;
    padding: 15px;
    border-radius: 6px;
    font-size: 12px;
    z-index: 10;
    max-width: 250px;
}

.controlInstructions h3 {
    margin: 0 0 10px 0;
    color: #2ecc40;
}

.controlInstructions ul {
    margin: 0;
    padding-left: 20px;
}

.controlInstructions li {
    margin: 5px 0;
}

/* Loading animation for video */
#robotVideo:not([srcObject]) {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200% 100%;
    animation: loading 1.5s infinite;
}

@keyframes loading {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/*********************************************/
/*************** DEVICE SIZING ***************/
/*********************************************/

/********** LARGE DEVICES **********/

@media only screen and (min-width: 1025px) { /* detect if screen is desktop */

    /***** hide mobile controls on desktop *****/

    #mobileControls {
        display: none;
    }

    /***** show desktop controls *****/

    .controlInstructions {
        display: block;
    }
}

/********** MEDIUM DEVICES **********/

@media only screen and (min-width: 501px) and (max-width: 1024px) { /* detect if screen is tablet */

    /***** force phone controls on tablet *****/

    .controlInstructions {
        display: none;
    }

    /***** mobile touch controls *****/

    #mobileControls {
        position: absolute;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 10;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 10px;
    }

    .controlRow {
        display: flex;
        gap: 15px;
        align-items: center;
        justify-content: center;
        height: fit-content;
    }

    .controlBtn {
        width: 60px;
        height: 60px;
        border: 2px solid #ffffff;
        border-radius: 50%;
        background: rgba(0, 0, 0, 0.7);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s ease;
        user-select: none;
        -webkit-user-select: none;
        -webkit-touch-callout: none;
    }

    .controlBtn:hover {
        background: #ffffff;
        border-color: #ffffff;
        transform: scale(1.1);
    }

    .controlBtn:active {
        background: #ffffff;
        transform: scale(0.95);
    }

    .controlBtn img {
        width: 30px;
        height: 30px;
        filter: brightness(0) invert(1);
    }

    .controlBtn span {
        font-size: 12px;
        font-weight: bold;
        font-family: 'Courier New', monospace;
    }

    .exitBtn {
        background: rgba(255, 65, 54, 0.7) !important;
        border-color: #ff4136 !important;
    }

    .exitBtn:hover {
        background: rgba(255, 65, 54, 0.9) !important;
    }

    /***** adjust video container for mobile *****/

    #videoContainer {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #000;
        overflow: hidden;
    }

    #robotVideo, #videoContainer canvas {
        width: 100vw;
        height: 100vh;
        object-fit: cover;
    }

    /***** adjust status elements for mobile *****/

    #connectButton {
        top: calc(var(--navBarHeight) + 15px);
        right: 20px;
        padding: 10px 20px;
        font-size: 14px;
    }

    #leaveButton {
        top: calc(var(--navBarHeight) + 15px);
        right: 20px;
        padding: 10px 20px;
        font-size: 14px;
    }

    /***** force mobile 8-button controls on tablet *****/

    #mobileControls8 {
        position: absolute;
        width: 100vw;
        height: 100vh;
        top: 0;
        left: 0;
        display: flex;
        flex-direction: row;
        justify-content: space-between;
        pointer-events: none;
        z-index: 10;
    }

    .mobileControlsLeft, .mobileControlsRight {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        align-items: center;
        height: fit-content;
        width: fit-content;
        pointer-events: none;
    }

    .mobileControlsLeft {
        align-items: flex-start;
    }

    .mobileControlsRight {
        align-items: flex-end;
    }

    .controlRowUpDown {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
    }

    .controlRowLRD {
        display: flex;
        flex-direction: row;
        justify-content: center;
        align-items: center;
        gap: 0.5em;
        width: 100%;
        margin-top: 0;
    }

    .controlBtn {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        padding: 0;
        margin: 0 6px;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        pointer-events: auto;
        touch-action: manipulation;
    }

    .controlBtn img {
        width: 40px;
        height: 40px;
        filter: brightness(0) invert(1);
        pointer-events: none;
        user-select: none;
    }

    .controlBtn:active img, .controlBtn:focus img {
        filter: brightness(0.7) invert(1) drop-shadow(0 0 8px #ffffff);
    }

    /* Hide any text in buttons */
    .controlBtn span { display: none; }

    /* Make sure controls are above video/status */
    #mobileControls8 { pointer-events: none; }
    .controlBtn { pointer-events: auto; }

    /* Position left controls above status, right controls above connect */
    .mobileControlsLeft {
        position: absolute;
        width: fit-content;
        z-index: 12;
        align-items: flex-start;
    }

    .mobileControlsRight {
        position: absolute;
        width: fit-content;
        z-index: 12;
        align-items: flex-end;
    }

    /* Transparent navBarOptions and projectsBarOptions, but not their children */
    .navBarOptions, .projectsBarOptions {
        opacity: 0 !important;
        background: transparent !important;
    }
    .navBarOptions > *, .projectsBarOptions > * {
        opacity: 1 !important;
    }

    /* Landscape overlay */
    #landscapeOverlay {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(0,0,0,0.7);
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .landscapeMessage {
        color: #fff;
        font-size: 1.5em;
        background: rgba(0,0,0,0.5);
        border-radius: 16px;
        padding: 32px 24px;
        text-align: center;
        max-width: 80vw;
        box-shadow: 0 2px 16px #0008;
    }
}

/********** SMALL DEVICES **********/

@media only screen and (max-width: 500px) { /* detect if screen is smartphone */

    /***** hide desktop controls on smartphone *****/

    .controlInstructions {
        display: none;
    }

    /***** mobile touch controls for smartphone *****/

    #mobileControls {
        position: absolute;
        bottom: 15px;
        left: 50%;
        transform: translateX(-50%);
        z-index: 10;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
    }

    .controlRow {
        display: flex;
        gap: 15px;
        align-items: center;
        justify-content: center;
        height: fit-content;
    }

    .controlBtn {
        width: 50px;
        height: 50px;
        border: 2px solid #ffffff;
        border-radius: 50%;
        background: rgba(0, 0, 0, 0.7);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s ease;
        user-select: none;
        -webkit-user-select: none;
        -webkit-touch-callout: none;
    }

    .controlBtn:hover {
        background: #ffffff;
        border-color: #ffffff;
        transform: scale(1.1);
    }

    .controlBtn:active {
        background: #ffffff;
        transform: scale(0.95);
    }

    .controlBtn img {
        width: 25px;
        height: 25px;
        filter: brightness(0) invert(1);
    }

    .controlBtn span {
        font-size: 10px;
        font-weight: bold;
        font-family: 'Courier New', monospace;
    }

    .exitBtn {
        background: rgba(255, 65, 54, 0.7) !important;
        border-color: #ff4136 !important;
    }

    .exitBtn:hover {
        background: rgba(255, 65, 54, 0.9) !important;
    }

    /***** adjust video container for smartphone *****/

    #videoContainer {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        z-index: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #000;
        overflow: hidden;
    }

    #robotVideo, #videoContainer canvas {
        width: 100vw;
        height: 100vh;
        object-fit: cover;
    }

    /***** adjust status elements for smartphone *****/

    #connectButton {
        top: calc(var(--navBarHeight) + 15px);
        right: 15px;
        padding: 8px 16px;
        font-size: 12px;
    }

    #leaveButton {
        top: calc(var(--navBarHeight) + 15px);
        right: 15px;
        padding: 8px 16px;
        font-size: 12px;
    }
}

/* --- Mobile 8-Button Controls --- */
@media only screen and (max-width: 1024px) {
  #mobileControls8 {
    position: absolute;
    width: 100vw;
    height: 100vh;
    top: 0;
    left: 0;
    display: flex;
    flex-direction: row;
    justify-content: space-between;
    pointer-events: none;
    z-index: 10;
  }
  .mobileControlsLeft, .mobileControlsRight {
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    align-items: center;
    height: 100%;
    width: fit-content;
    pointer-events: none;
  }
  .mobileControlsLeft {
    align-items: flex-start;
  }
  .mobileControlsRight {
    align-items: flex-end;
  }
  .controlRowUpDown {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
  }
  .controlRowLRD {
    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;
    width: 100%;
  }
  .controlBtn {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 0;
    margin: 0 6px;
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: auto;
    touch-action: manipulation;
  }
  .controlBtn img {
    width: 40px;
    height: 40px;
    filter: brightness(0) invert(1);
    pointer-events: none;
    user-select: none;
  }
  .controlBtn:active img, .controlBtn:focus img {
    filter: brightness(0.7) invert(1) drop-shadow(0 0 8px #ffffff);
  }
  /* Hide any text in buttons */
  .controlBtn span { display: none; }
  /* Make sure controls are above video/status */
  #mobileControls8 { pointer-events: none; }
  .controlBtn { pointer-events: auto; }
  /* Position left controls above status, right controls above connect */
  .mobileControlsLeft {
    position: absolute;
    width: fit-content;
    z-index: 12;
    align-items: flex-start;
  }
  .mobileControlsRight {
    position: absolute;
    width: fit-content;
    z-index: 12;
    align-items: flex-end;
  }

  /* Transparent navBarOptions and projectsBarOptions, but not their children */
  .navBarOptions, .projectsBarOptions {
    opacity: 0 !important;
    background: transparent !important;
  }
  .navBarOptions > *, .projectsBarOptions > * {
    opacity: 1 !important;
  }
  /* Landscape overlay */
  #landscapeOverlay {
    position: fixed;
    top: 0; left: 0; width: 100vw; height: 100vh;
    background: rgba(0,0,0,0.7);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .landscapeMessage {
    color: #fff;
    font-size: 1.5em;
    background: rgba(0,0,0,0.5);
    border-radius: 16px;
    padding: 32px 24px;
    text-align: center;
    max-width: 80vw;
    box-shadow: 0 2px 16px #0008;
  }
}

@media only screen and (max-width: 500px) {
  .controlBtn {
    width: 36px;
    height: 36px;
  }
  .controlBtn img {
    width: 30px;
    height: 30px;
  }
  .mobileControlsLeft, .mobileControlsRight {
    width: fit-content;
  }

  /* Smaller action and jump buttons for smartphones */
  .actionBtnRect, .jumpBtnRect {
    width: 60px;
    height: 36px;
    background: #fff;
    border: 2px solid #ffffff;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: background-color 0.3s, border-color 0.3s, transform 0.2s;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin: 12px 0 0 0;
    pointer-events: auto;
  }
  .actionBtnRect:hover, .jumpBtnRect:hover {
    background: #ffffff;
    color: #fff;
    border-color: #ffffff;
    transform: scale(1.05);
  }
  .actionBtnRect:active, .jumpBtnRect:active {
    background: #ffffff;
    color: #fff;
    border-color: #ffffff;
    transform: scale(0.95);
  }
}

@media only screen and (max-width: 1024px) {
  #navBar {
    background-color: rgba(52,58,64,0) !important;
    backdrop-filter: none;
  }
  #navBar * {
    opacity: 1 !important;
  }
  #navBarOptionsBox, #projectsBarOptionsBox {
    background-color: rgba(52,58,64,0) !important;
    box-shadow: none !important;
  }
  #navBarOptionsButton {
    background-color: rgba(52,58,64,0) !important;
    box-shadow: none !important;
  }
  #navBarOptionsButton .navBarOptionsLines {
    background-color: white !important;
    opacity: 1 !important;
  }
}

.mobileControlsLeft .actionBtnRect, .mobileControlsRight .jumpBtnRect {
  pointer-events: auto !important;
}

#actionBtn, #jumpBtn {
    background: #fff;
    border: 2px solid #ffffff;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    transition: background-color 0.3s, border-color 0.3s, transform 0.2s;
    width: 192px;
    height: 40px;
    margin-top: 8px;
}
#actionBtn:hover, #jumpBtn:hover {
  background: #ffffff;
  color: #fff;
  border-color: #ffffff;
  transform: scale(1.05);
}
#actionBtn:active, #jumpBtn:active {
  background: #ffffff;
  color: #fff;
  border-color: #ffffff;
  transform: scale(0.95);
}

/* Move mobileControlsLeft and mobileControlsRight to the bottom corners */
.mobileControlsLeft {
    position: absolute;
    left: 15px;
    bottom: 15px;
    width: fit-content;
    z-index: 12;
    align-items: flex-start;
}
.mobileControlsRight {
    position: absolute;
    right: 15px;
    bottom: 15px;
    width: fit-content;
    z-index: 12;
    align-items: flex-end;
}
