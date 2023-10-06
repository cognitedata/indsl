import { storiesOf } from "@storybook/react";
import { defaultTranslations } from "components/NodeEditor/translations";
import InfoModal from "components/NodeEditor/V2/Nodes/ActionBar/InfoModal";
import ReactFlowNodeEditor from "components/NodeEditor/V2/ReactFlowNodeEditor";
import { NodeTypes } from "components/NodeEditor/V2/types";
import response from "./assets/response.json";

response.forEach((indslFunction) =>
  indslFunction.versions.forEach((version) =>
    storiesOf(
      `${indslFunction.category}/${version.name}/v${version.version}`,
      module
    )
      .add(`${version.name} v${version.version} Info Modal`, () => (
        // @ts-ignore
        <InfoModal isOpen indslFunction={version} />
      ))
      .add(`${version.name} v${version.version} Block`, () => (
        <ReactFlowNodeEditor
          sources={[]}
          operations={[]}
          settings={{ autoAlign: true }}
          isValid
          onSaveSettings={() => {}}
          onElementsRemove={() => {}}
          onConnect={() => {}}
          onEdgeUpdate={() => {}}
          onNodeDragStop={() => {}}
          onAddSourceNode={() => {}}
          onAddConstantNode={() => {}}
          onAddFunctionNode={() => {}}
          onAddOutputNode={() => {}}
          onMove={() => {}}
          flowElements={[
            {
              id: "sample",
              data: {
                selectedOperation: {
                  op: indslFunction.op,
                  version: version.version,
                },
                // @ts-ignore
                operation: indslFunction,
                parameterValues: {},
                onParameterValuesChange: () => {},
                onDuplicateNode: () => {},
                onRemoveNode: () => {},
                readOnly: false,
                translations: defaultTranslations,
              },
              type: NodeTypes.FUNCTION,
              position: { x: 50, y: 100 },
            },
          ]}
          translations={defaultTranslations}
        />
      ))
      .add(`${version.name} v${version.version} Parameters`, () => (
        <ReactFlowNodeEditor
          sources={[]}
          // @ts-ignore
          operations={response}
          settings={{ autoAlign: true }}
          isValid
          onSaveSettings={() => {}}
          onElementsRemove={() => {}}
          onConnect={() => {}}
          onEdgeUpdate={() => {}}
          onNodeDragStop={() => {}}
          onAddSourceNode={() => {}}
          onAddConstantNode={() => {}}
          onAddFunctionNode={() => {}}
          onAddOutputNode={() => {}}
          onMove={() => {}}
          translations={defaultTranslations}
          flowElements={[
            {
              id: "sample",
              data: {
                selectedOperation: {
                  op: indslFunction.op,
                  version: version.version,
                },
                // @ts-ignore
                operation: indslFunction,
                parameterValues: {},
                onParameterValuesChange: () => {},
                onDuplicateNode: () => {},
                onRemoveNode: () => {},
                readOnly: false,
                translations: defaultTranslations,
                formOpen: true,
              },
              type: NodeTypes.FUNCTION,
              position: { x: 50, y: 100 },
            },
          ]}
        />
      ))
  )
);
