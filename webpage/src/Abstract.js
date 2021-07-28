import React from "react";
import { Container, Row, Col } from "react-bootstrap";

const Abstract = (props) => {

    return (
        <Container style={{paddingTop: 40, width: "100%"}} className="justify-content-md-left" fluid>
            <Row align="center">
                <img src={require("" + props.img).default} style={{width: "80%", margin: "0 auto"}}/>
            </Row>
            <br/>
            <Row align="center">
                <p align="justify" style={{width: "80%", margin: "0 auto"}}><b>tl;dr: </b>{props.tldr}</p>
            </Row>
            <br/>
            <Row align="center">
                <p align="justify" style={{width: "80%", margin: "0 auto"}}><b>Abstract: </b>{props.abstract}</p>
            </Row>
        </Container>
    );
}

export default  Abstract;
