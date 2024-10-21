import React from 'react'
import './orb.css'
import { useWindowSize } from '../../Hooks/useWindowSize'


function Orb() {
    const { width, height } = useWindowSize()
    return (
        <div classsName="Orb-div">
        <style>
        {`
            @keyframes moveorb{
                0%{
                 transform: translate(0, 0);
                }
                50%{
                 transform: translate(${width/1.2}px, ${height/2}px);
                }
                100%{
                 transform: translate(0, 0);
                }
            }
        `}
        </style>
        </div>
    )
}
export default Orb